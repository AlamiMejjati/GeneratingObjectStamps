import argparse
import glob
import numpy as np
from six.moves import range
from dataloader import *
from tensorpack import *
from tensorpack.utils import logger
from tensorpack.tfutils.summary import add_moving_summary
from tensorpack.utils.gpu import get_num_gpu
from tensorpack.tfutils.scope_utils import auto_reuse_variable_scope
import tensorflow as tf
from tensorflow.python.training import moving_averages
from tensorpack.tfutils.export import ModelExporter
from GAN import GANTrainer, GANModelDesc
import os
import sys
import six
from datetime import datetime


DIS_SCALE = 3
SHAPE = 256
BATCH = 4
TEST_BATCH = 8
NF = 64  # channel size
N_DIS = 4
N_SAMPLE = 2
N_SCALE = 3
N_RES = 4
STYLE_DIM = 128
STYLE_DIM_z2 = 8
n_upsampling = 5
chs = NF * 8
nb_blocks = 9
enable_argscope_for_module(tf.layers)

def SmartInit(obj, ignore_mismatch=False):
    """
    Create a :class:`SessionInit` to be loaded to a session,
    automatically from any supported objects, with some smart heuristics.
    The object can be:

    + A TF checkpoint
    + A dict of numpy arrays
    + A npz file, to be interpreted as a dict
    + An empty string or None, in which case the sessinit will be a no-op
    + A list of supported objects, to be initialized one by one

    Args:
        obj: a supported object
        ignore_mismatch (bool): ignore failures when the value and the
            variable does not match in their shapes.
            If False, it will throw exception on such errors.
            If True, it will only print a warning.

    Returns:
        SessionInit:
    """
    if not obj:
        return JustCurrentSession()
    if isinstance(obj, list):
        return ChainInit([SmartInit(x, ignore_mismatch=ignore_mismatch) for x in obj])
    if isinstance(obj, six.string_types):
        obj = os.path.expanduser(obj)
        if obj.endswith(".npy") or obj.endswith(".npz"):
            assert tf.gfile.Exists(obj), "File {} does not exist!".format(obj)
            filename = obj
            logger.info("Loading dictionary from {} ...".format(filename))
            if filename.endswith('.npy'):
                obj = np.load(filename, encoding='latin1').item()
            elif filename.endswith('.npz'):
                obj = dict(np.load(filename))
        elif len(tf.gfile.Glob(obj + "*")):
            # Assume to be a TF checkpoint.
            # A TF checkpoint must be a prefix of an actual file.
            return (SaverRestoreRelaxed if ignore_mismatch else SaverRestore)(obj)
        else:
            raise ValueError("Invalid argument to SmartInit: " + obj)

    if isinstance(obj, dict):
        return DictRestore(obj)
    raise ValueError("Invalid argument to SmartInit: " + type(obj))

def tpad(x, pad, mode='CONSTANT',  name=None):
    return tf.pad(x, [[0, 0], [pad, pad], [pad, pad], [0, 0]], mode=mode)

def AdaIN(x, gamma=1.0, beta=0, epsilon=1e-5):
    # gamma, beta = style_mean, style_std from MLP

    #c_mean, c_var = tf.nn.moments(x, axes=[1, 2], keep_dims=True)
    #c_std = tf.sqrt(c_var + epsilon)

    x = InstanceNorm('inorm', x, use_affine=False)
    return tf.add(gamma * x, beta, name='AdaIn')

def AdaINReLU(x, gamma=1.0, beta=0.0, name=None):
    x = AdaIN(x, gamma=gamma, beta=beta)
    return tf.nn.relu(x, name=name)

def LNLReLU(x, name=None):
    x = LayerNorm('lnorm', x)
    return tf.nn.leaky_relu(x, name=name)

def instance_norm(x):

    with tf.variable_scope("instance_norm"):
        epsilon = 1e-5
        mean, var = tf.nn.moments(x, [1, 2], keepdims=True)
        scale = tf.get_variable('scale', [x.get_shape()[-1]],
                                initializer=tf.truncated_normal_initializer(
                                    mean=1.0, stddev=0.02
        ))
        offset = tf.get_variable(
            'offset', [x.get_shape()[-1]],
            initializer=tf.constant_initializer(0.0)
        )
        out = scale * tf.div(x - mean, tf.sqrt(var + epsilon)) + offset

        return out

def AdaLN(x, gamma=1.0, beta=0):
    return tf.add(gamma * LayerNorm('lnorm', x), beta, name='AdaLN')

def GroupNorm(x, G=32, epsilon=1e-5, scope=None):
    with tf.variable_scope(scope, 'GroupNorm'):
        _, H, W, C = x.get_shape().as_list()
        G = min(G, C)

        x = tf.reshape(x, [-1, H, W, G, C // G])
        mean, var = tf.nn.moments(x, [1, 2, 4], keep_dims=True)
        x = (x - mean) * tf.rsqrt(var + epsilon)

        gamma = tf.get_variable('gamma', [1, 1, 1, C], initializer=tf.constant_initializer(1.0))
        beta = tf.get_variable('beta', [1, 1, 1, C], initializer=tf.constant_initializer(0.0))


        x = tf.reshape(x, [-1, H, W, C]) * gamma + beta
    return x

def GNLReLU(x, name=None):
    x = GroupNorm(x)
    return tf.nn.leaky_relu(x, name=name)

def INLReLU(x, name=None):
    x = InstanceNorm('in', x)
    return tf.nn.leaky_relu(x, name=name)

def up_sample(x, scale_factor=2):
    _, h, w, _ = x.get_shape().as_list()
    new_size = [h * scale_factor, w * scale_factor]
    return tf.image.resize_nearest_neighbor(x, size=new_size,
            align_corners=True)

def adaptive_avg_pooling(x):
    # global average pooling
    gap = tf.reduce_mean(x, axis=[1, 2], keepdims=True)
    return gap

class CheckNumerics(Callback):
    """
    When triggered, check variables in the graph for NaN and Inf.
    Raise exceptions if such an error is found.
    """
    def _setup_graph(self):
        vars = tf.trainable_variables()
        ops = [tf.check_numerics(v, "CheckNumerics['{}']".format(v.op.name)).op for v in vars]
        self._check_op = tf.group(*ops)

    def _before_run(self, _):
        self._check_op.run()


class Model(GANModelDesc):
    def inputs(self):
        return [tf.placeholder(tf.float32, (None, SHAPE, SHAPE, 3), 'input'),
                tf.placeholder(tf.float32, (None, SHAPE, SHAPE, 1), 'template'),
                tf.placeholder(tf.float32, (None, SHAPE, SHAPE, 1), 'mask'),
                tf.placeholder(tf.float32, (None, 1, 1, 4), 'bbx'),
                tf.placeholder(tf.float32, (None, 1, 1, STYLE_DIM), 'z1'),
                tf.placeholder(tf.float32, (None, 1, 1, STYLE_DIM_z2), 'z2'),
               ]

    @staticmethod
    def build_res_block(x, name, chan, first=False):
        with tf.variable_scope(name), \
            argscope([Conv2D], kernel_size=3, strides=1):
            input = x
            x =(LinearWrap(x)
                    .tf.pad([[0, 0], [1, 1], [1, 1], [0, 0]], mode='SYMMETRIC')
                    .Conv2D('conv0', chan, 3, padding='VALID')
                    .tf.pad([[0, 0], [1, 1], [1, 1], [0, 0]], mode='SYMMETRIC')
                    .Conv2D('conv1', chan, 3, padding='VALID', activation=tf.identity)())
            return GroupNorm(x) + input

    @staticmethod
    def build_AdaIN_res_block(x, mu, sigma, name, chan, first=False):
        with tf.variable_scope(name), \
                argscope([tf.layers.conv2d], kernel_size=3, strides=1, padding='VALID'):
            input = x
            activ = lambda y: tf.add(mu * GroupNorm(y), sigma, name='gnorm')
            #activ = lambda y: InstanceNorm('inorm', y)
            x = tpad(x, pad=1, mode='SYMMETRIC')
            x = tf.layers.conv2d(x, chan, activation=activ, name='conv0')
            x = tf.nn.leaky_relu(x)
            x = tpad(x, pad=1, mode='SYMMETRIC')
            x = tf.layers.conv2d(x, chan, activation=activ, name='conv1')
            return x + input

    @auto_reuse_variable_scope
    def style_generator(self, content, style):
        l = content
        channel = pow(2, N_SAMPLE) * NF
        with tf.variable_scope('igen'):
            mu, sigma = self.MLP(style)

            #l = (LinearWrap(img)
            #     .tf.pad([[0, 0], [0, 0], [3, 3], [3, 3]], mode='SYMMETRIC')
            #     .Conv2D('conv0', NF, 7, padding='VALID')
            #     .Conv
            for k in range(N_RES):
                l = Model.build_AdaIN_res_block(l, 'res{}'.format(k), channel,
                        mu, sigma, first=(k == 0))

        return l

    @auto_reuse_variable_scope
    def decoder(self, c, s, res_shape):
        channel = pow(2, N_SAMPLE) * NF + NF*2
        with tf.variable_scope('dec'), argscope([Conv2D, Conv2DTranspose], activation=GNLReLU):
            # s = tf.layers.dense(s, np.prod(res_shape), name='fc1', activation=tf.nn.leaky_relu)
            # s = tf.reshape(s, [-1] + res_shape)
            s = (LinearWrap(s)
                .Conv2DTranspose('deconv0s', NF*2, 3, strides=2)
                .Conv2DTranspose('deconv1s', NF*2, 3, strides=2)
                .Conv2DTranspose('deconv2s', NF*2, 3, strides=2)())
            res_in = tf.concat([c, s], axis=-1)

            for k in range(N_RES):
                l = Model.build_res_block(res_in, 'res{}'.format(k), channel, first=(k == 0))

            l = (LinearWrap(l)
                .Conv2DTranspose('deconv0', NF * 2, 3, strides=2)
                .Conv2DTranspose('deconv1', NF * 1, 3, strides=2)
                .tf.pad([[0, 0], [3, 3], [3, 3], [0, 0]], mode='SYMMETRIC')
                .Conv2D('convlast', 3, 7, padding='VALID', activation=tf.tanh, use_bias=True)())
        return l

    @auto_reuse_variable_scope
    def style_encoder(self, x):
        chan = NF
        with tf.variable_scope('senc'), \
            argscope([tf.layers.conv2d, Conv2D],
                    activation=GNLReLU, kernel_size=4, strides=2):

            x = tpad(x, pad=3, mode='reflect')
            x = tf.layers.conv2d(x, chan, kernel_size=7, strides=1,
                        name='conv_0', activation=tf.nn.leaky_relu)

            for i in range(2):
                x = tpad(x, pad=1, mode='reflect')
                x = tf.layers.conv2d(x, chan*2, name='conv_%d' % (i+1))
                chan*=2

            for i in range(2):
                x = tpad(x, pad=1, mode='reflect')
                x = tf.layers.conv2d(x, chan, name='dconv_%d' % i)

            x = adaptive_avg_pooling(x) # global average pooling
            x = tf.layers.conv2d(x, STYLE_DIM_z2, kernel_size=1, strides=1,
                activation=None, name='SE_logit')
        return x

    @auto_reuse_variable_scope
    def content_encoder(self, x):
        chan = NF
        with tf.variable_scope('cenc'), argscope([tf.layers.conv2d],
                activation=GNLReLU):

            x = tpad(x, pad=3, mode='reflect')
            x = tf.layers.conv2d(x, chan, kernel_size=7, strides=1,
                    name='conv_0', activation=tf.nn.leaky_relu)

            for i in range(N_SAMPLE):
                x = tpad(x, 1, mode='reflect')
                x = tf.layers.conv2d(x, chan*2, kernel_size=4, strides=2,
                        name='conv_%d' % (i + 1))
                chan*=2

            for i in range(N_RES):
                #x = tpad(x, 1, mode'reflect')
                x = Model.build_res_block(x, 'res%d' % i, chan, first=(i==0))
        return x

    @auto_reuse_variable_scope
    def MLP(self, style, channel, nb_upsampling,  name='MLP'):
        # channel = pow(2, N_SAMPLE) * NF
        with tf.variable_scope(name), \
                argscope([tf.layers.dense],
                    kernel_initializer=tf.contrib.layers.variance_scaling_initializer()):

            style = tf.layers.flatten(style)
            x = tf.layers.dense(style, channel*2, activation=tf.nn.leaky_relu, name='linear_0')
            x = tf.layers.dense(x, channel*2, activation=tf.nn.leaky_relu, name='linear_1')
            x = tf.layers.dense(x, channel * 2, activation=tf.nn.leaky_relu, name='linear_2')

            x = tf.tile(x[:, np.newaxis], [1, nb_upsampling, 1])
            # mu = x[:, :channel]
            # sigma = x[:, channel:]
            # 
            # mu = tf.reshape(mu, shape=[-1, 1, 1, channel])
            # sigma = tf.reshape(sigma, shape=[-1, 1, 1, channel])
            #mu = tf.split(mu, N_RES, 0)
            #sigma = tf.split(sigma, N_RES, 0)
        return x

    @auto_reuse_variable_scope
    def zdec(self, box, style, reshape,name='zdec'):
        # zshape = reshape[:]
        # zshape[-1] = 28
        # channel = np.prod(zshape)#8*8*NF
        with tf.variable_scope(name), \
                argscope([Conv2D, tf.layers.dense], activation=GNLReLU,
                    kernel_initializer=tf.contrib.layers.variance_scaling_initializer()):

            mu, sigma = self.MLP(style, channel=reshape[-1])
            # l = tf.layers.dense(style, channel,
            #         activation=tf.identity, name='linear_0')
            # l = tf.reshape(l, [-1] + zshape)
            # l = tf.nn.leaky_relu(GroupNorm(l, G=1))
            # l = GNLReLU(l)

            x = (LinearWrap(box)
                .tf.pad([[0, 0], [3, 3], [3, 3], [0, 0]], mode='SYMMETRIC')
                .Conv2D('conv0', reshape[-1], 7, strides=1, padding='VALID')
                .Conv2D('conv1', reshape[-1], 3, strides=2)
                .Conv2D('conv2', reshape[-1], 3, strides=2)
                .Conv2D('conv3', reshape[-1], 3, strides=2)
                .Conv2D('conv4', reshape[-1], 3, strides=2)
                .Conv2D('conv5', reshape[-1], 3, strides=2)())

            # lx = tf.concat([x,l], axis=-1)

            # for k in range(N_RES):
            #     l = Model.build_res_block(x, 'res{}'.format(k), reshape[-1], first=(k == 0))

            for k in range(N_RES):
                l = Model.build_AdaIN_res_block(x, mu, sigma, 'res_adain_{}'.format(k), reshape[-1], first=(k == 0))

        return l

    @auto_reuse_variable_scope
    def encoder(self, img, chan=3):
        assert img is not None
        with argscope([Conv2D, Conv2DTranspose], activation=GNLReLU):
            l = (LinearWrap(img)
                 .tf.pad([[0, 0], [3, 3], [3, 3], [0, 0]], mode='SYMMETRIC')
                 .Conv2D('conv0', NF, 7, padding='VALID')
                 .Conv2D('conv1', NF * 2, 3, strides=2)
                 .Conv2D('conv2', NF * 4, 3, strides=2)())
            for k in range(N_RES * 2):
                l = Model.build_res_block(l, 'res{}'.format(k), NF * 4, first=(k == 0))
        return l

    @auto_reuse_variable_scope
    def encoder_mask(self, img, chan=3):
        assert img is not None
        with tf.variable_scope('menc'), argscope([Conv2D, Conv2DTranspose], activation=GNLReLU):
            l = (LinearWrap(img)
                 .tf.pad([[0, 0], [3, 3], [3, 3], [0, 0]], mode='SYMMETRIC')
                 .Conv2D('conv0', NF, 7, padding='VALID')
                 .Conv2D('conv1', NF*2, 3, strides=2)
                 .Conv2D('conv2', NF*2, 3, strides=2)
                 .Conv2D('conv3', NF*2, 3, strides=2)
                 .Conv2D('conv4', NF*2, 3, strides=2)
                 .Conv2D('conv5', NF*2, 3, strides=2)())
            for k in range(N_RES):
                l = Model.build_res_block(l, 'res{}'.format(k), NF*2, first=(k == 0))

            res_shape = l.get_shape().as_list()[1:]
            # l =tf.layers.flatten(l)
            # l = tf.layers.dense(l, STYLE_DIM, name='fc_mask')
        return l, res_shape

    @auto_reuse_variable_scope
    def decoder_(self, img, chan=3):
        assert img is not None
        with argscope([Conv2D, Conv2DTranspose], activation=GNLReLU):
            l = (LinearWrap(img)
                 .Conv2DTranspose('deconv0', NF * 2, 3, strides=2)
                 .Conv2DTranspose('deconv1', NF * 1, 3, strides=2)
                 .tf.pad([[0, 0], [3, 3], [3, 3], [0, 0]], mode='SYMMETRIC')
                 .Conv2D('convlast', chan, 7, padding='VALID', activation=tf.tanh, use_bias=True)())
        return l

    @auto_reuse_variable_scope
    def generator(self, img, chan=3):
        assert img is not None
        with argscope([Conv2D, Conv2DTranspose], activation=GNLReLU):
            l = (LinearWrap(img)
                 .tf.pad([[0, 0], [3, 3], [3, 3], [0, 0]], mode='SYMMETRIC')
                 .Conv2D('conv0', NF, 7, padding='VALID')
                 .Conv2D('conv1', NF * 2, 3, strides=2)
                 .Conv2D('conv2', NF * 4, 3, strides=2)())
            for k in range(9):
                l = Model.build_res_block(l, 'res{}'.format(k), NF * 4, first=(k == 0))
            l = (LinearWrap(l)
                 .Conv2DTranspose('deconv0', NF * 2, 3, strides=2)
                 .Conv2DTranspose('deconv1', NF * 1, 3, strides=2)
                 .tf.pad([[0, 0], [3, 3], [3, 3], [0, 0]], mode='SYMMETRIC')
                 .Conv2D('convlast', chan, 7, padding='VALID', activation=tf.tanh, use_bias=True)())
        return l

    @auto_reuse_variable_scope
    def generator_adain(self, img, musigma, nb_blocks=9, chan=3):
        assert img is not None
        with argscope([Conv2D, Conv2DTranspose], activation=INLReLU):
            l = (LinearWrap(img)
                 .tf.pad([[0, 0], [3, 3], [3, 3], [0, 0]], mode='SYMMETRIC')
                 .Conv2D('conv0', NF, 7, padding='VALID')
                 .Conv2D('conv1', NF * 4, 3, strides=2)
                 .Conv2D('conv2', NF * 8, 3, strides=2)())

            for k in range(nb_blocks):
                musigmak = musigma[:,k]
                l = Model.build_adain_res_block(l, musigmak, 'res{}'.format(k), NF * 8, first=(k == 0))

            l = (LinearWrap(l)
                 .Conv2DTranspose('deconv0', NF * 8, 3, strides=2)
                 .Conv2DTranspose('deconv1', NF * 4, 3, strides=2)
                 .tf.pad([[0, 0], [3, 3], [3, 3], [0, 0]], mode='SYMMETRIC')
                 .Conv2D('convlast', chan, 7, padding='VALID', activation=tf.tanh, use_bias=True)())
        return l

    @auto_reuse_variable_scope
    def sr_net(self, m, chan=1):
        assert m is not None
        with argscope([Conv2D, Conv2DTranspose], activation=INLReLU):
            m = tf.keras.layers.UpSampling2D(2, data_format=None)(m)
            l = (LinearWrap(m)
                 .Conv2D('conv0_sr', NF, 7, padding='SAME')
                 .Conv2D('conv1_sr', NF, 3, padding='SAME')
                 .Conv2D('conv2_sr', chan, 7, padding='SAME', activation=tf.tanh, use_bias=True)())
        return l

    @staticmethod
    def build_adain_res_block(x, musigma, name, chan, first=False):
        with tf.variable_scope(name), \
            argscope([Conv2D], kernel_size=3, strides=1):

            musigma = tf.reshape(musigma, [-1, 2, chan])
            mu = musigma[:, 0]
            mu = tf.reshape(mu, [-1, 1, 1, chan])
            sigma = musigma[:, 1]
            sigma = tf.reshape(sigma, [-1, 1, 1, chan])

            input = x
            x = Conv2D('conv0', x, chan, 3, activation=tf.nn.leaky_relu, strides=1)
            x = tf.add(mu * InstanceNorm('in_0', x, use_affine=False), sigma, name='adain_0')

            x = Conv2D('conv1', x, chan, 3, activation=tf.nn.leaky_relu, strides=1)
            x = tf.add(mu * InstanceNorm('in_1', x, use_affine=False), sigma, name='adain_1')

            return x+input

            # x =(LinearWrap(x)
            #         .tf.pad([[0, 0], [1, 1], [1, 1], [0, 0]], mode='SYMMETRIC')
            #         .Conv2D('conv0', chan, 3, padding='VALID')
            #         .tf.pad([[0, 0], [1, 1], [1, 1], [0, 0]], mode='SYMMETRIC')
            #         .Conv2D('conv1', chan, 3, padding='VALID', activation=tf.identity)())
            # return GroupNorm(x) + input

    @auto_reuse_variable_scope
    def gen_from_zbox_adain(self, z, musigma, chan=3, n_upsampling=5):
        with argscope([Conv2D, Conv2DTranspose]):

            x = Conv2DTranspose('deconvInit', z, chs, 1, activation=tf.nn.leaky_relu, strides=4)

            for i in range(n_upsampling):
                x = Conv2D('conv%d'%i, x, chs, 3, activation=tf.nn.leaky_relu, strides=1)
                musigmai = musigma[:,i]
                musigmai = tf.reshape(musigmai, [-1,2,chs])
                mu = musigmai[:, 0]
                mu = tf.reshape(mu, [-1,1,1,chs])
                sigma = musigmai[:, 1]
                sigma = tf.reshape(sigma, [-1, 1, 1, chs])
                x = tf.add(mu * InstanceNorm('in_%i'%i, x, use_affine=False), sigma, name='adain%d'%i)
                x = Conv2DTranspose('deconv%d'%i, x, chs, 3 ,activation=tf.nn.leaky_relu, strides=2) #8*8

            x = Conv2D('conv%d' % (i+1), x, NF * 4, 3, activation=tf.nn.leaky_relu, strides=1)
            x = InstanceNorm('in_%i' % (i + 1), x, use_affine=True)
            x = Conv2D('conv%d' % (i+2), x, NF * 2, 3, activation=tf.nn.leaky_relu, strides=1)
            x = InstanceNorm('in_%i' % (i + 2), x, use_affine=True)
            x = Conv2D('conv%d' % (i+3), x, 1, 3, activation=tf.nn.tanh, strides=1)
        return x

    @auto_reuse_variable_scope
    def z_reconstructer(self, musigma, dimz ,name='z_reconstructer'):
        with tf.variable_scope(name), \
             argscope([tf.layers.dense],
                      kernel_initializer=tf.contrib.layers.variance_scaling_initializer()):
            musigma = tf.layers.flatten(musigma)
            x = tf.layers.dense(musigma, dimz, activation=tf.nn.leaky_relu, name='linear_0')
            x = tf.layers.dense(x, dimz, activation=tf.nn.leaky_relu, name='linear_1')
            x=tf.reshape(x, [-1,1,1,dimz])
        return x

    @auto_reuse_variable_scope
    def discrim_enc(self, img):
        with argscope(Conv2D, activation=INLReLU, kernel_size=4, strides=2):
            l1 = Conv2D('conv0', img, NF, activation=tf.nn.leaky_relu)
            l2 = Conv2D('conv1', l1, NF * 2)
            features = Conv2D('conv2', l2, NF * 4)
        return features, [l1, l2]

    @auto_reuse_variable_scope
    def discrim_classify(self, img):
        with argscope(Conv2D, activation=INLReLU, kernel_size=4, strides=2):
            l1 = Conv2D('conv3', img, NF * 4, strides=2)
            l2 = Conv2D('conv4', l1, NF * 4, strides=1)
            l2 = tf.layers.flatten(l2)
            l3 = tf.layers.dense(l2, 1, activation=tf.identity, name='imisreal')
            return l3, [l1]

    @auto_reuse_variable_scope
    def discrim_patch_classify(self, img):
        with argscope(Conv2D, activation=INLReLU, kernel_size=4, strides=2):
            l1 = Conv2D('conv3', img, NF * 8, strides=1)
            l2 = Conv2D('conv4', l1, 1, strides=1, activation=tf.identity, use_bias=True)
            return l2, [l1]
    @auto_reuse_variable_scope
    def discriminator(self, img):
        with argscope(Conv2D, activation=GNLReLU, kernel_size=4, strides=1):
            features = (LinearWrap(img)
                 .Conv2D('conv0', NF, activation=tf.nn.leaky_relu ,strides=2)
                 .Conv2D('conv1', NF * 2)
                 .Conv2D('conv2', NF * 4)())
            l = (LinearWrap(features)
                 .Conv2D('conv3', NF * 8, strides=1)
                 .Conv2D('conv4', 1, strides=1, activation=tf.identity, use_bias=True)())
        return l, features

    def get_feature_match_loss(self, feats_real, feats_fake, name):
        losses = []

        for i, (real, fake) in enumerate(zip(feats_real, feats_fake)):
            with tf.variable_scope(name):
                fm_loss_real = tf.get_variable('fm_real_%d' % i,
                        real.shape[1:],
                        dtype=tf.float32,
                    #expected_shape=real.shape[1:],
                        trainable=False)

                ema_real_op = moving_averages.assign_moving_average(fm_loss_real,
                    tf.reduce_mean(real, 0), 0.99, zero_debias=False,
                    name='EMA_fm_real_%d' % i)

            loss = tf.reduce_mean(tf.squared_difference(
                fm_loss_real,
                tf.reduce_mean(fake, 0)),
                name='mse_feat_' + real.op.name)

            losses.append(loss)

            tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, ema_real_op)
            #tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, ema_fake_op)

        ret = tf.add_n(losses, name='feature_match_loss')
        #add_moving_summary(ret)
        return ret

    def build_graph(self, img, box, mask, bbx, z, z2):

        with tf.name_scope('preprocess'):
            img_crop = tf.multiply(img, 1 - box)
            img = (img / 128.0 - 1.0)
            img_crop = (img_crop / 128.0 - 1.0)
            bin_mask = mask/255.
            img_inter = ((img * bin_mask) - (1-bin_mask))*box + img*(1-box)
            mask = (mask / 128.0 - 1.0)

        with tf.name_scope('styleIn'):
            style_shape_z2 = [tf.shape(mask)[0], 1, 1, STYLE_DIM]
            z2 = tf.random_normal(style_shape_z2, mean=0.0, stddev=1.0, dtype=tf.float32, name='stylegtrand')

        def vizN(name, a):
            with tf.name_scope(name):
                m = tf.concat(a, axis=2)
                m = tf.image.grayscale_to_rgb(m)
                m = (m + 1.0) * 128
                m = tf.clip_by_value(m, 0, 255)
                m = tf.cast(m, tf.uint8, name='viz')
            tf.summary.image(name, m, max_outputs=50)

        # use the initializers from torch
        with argscope([Conv2D, Conv2DTranspose, tf.layers.conv2d]):
            #Let us encode the images
            with tf.variable_scope('gen'):
                mlp_in = tf.concat([z, bbx], axis=-1)
                with tf.variable_scope('mlpz'):
                    mu_sigma = self.MLP(mlp_in, chs, nb_blocks)
                with tf.variable_scope('genmask'):
                    gen_mask = self.generator_adain(img_crop, mu_sigma, nb_blocks, 1)
                with tf.variable_scope('recon_z'):
                    z_recon = self.z_reconstructer(tf.reduce_mean(mu_sigma, axis=1), STYLE_DIM)


            #The final discriminator that takes them both
            discrim_out_mask = []
            discrim_fm_real_mask = []
            discrim_fm_fake_mask = []

            with tf.variable_scope('discrim'):
                with tf.variable_scope('discrim_mask'):
                    def downsample(img):
                        return tf.layers.average_pooling2d(img, 3, 2)

                    # D_inputs = [img, tf.concat([intermediary_GT, mask], axis=-1),
                    #             recon_im, recon_inter]
                    bbx_tile = tf.tile(bbx, [1, SHAPE, SHAPE, 1])
                    D_input_real = tf.concat([mask, bbx_tile], axis=-1)
                    D_input_fake = tf.concat([gen_mask, bbx_tile], axis=-1)
                    D_inputs = [D_input_real, D_input_fake]

                    for s in range(DIS_SCALE):
                        with tf.variable_scope('s%d'%s):
                            if s != 0:
                                D_inputs = [downsample(im) for im in D_inputs]

                            mask_s, mask_recon_s = D_inputs

                            with tf.variable_scope('Ax'):
                                Ax_feats_real, Ax_fm_real = self.discrim_enc(mask_s)
                                Ax_feats_fake, Ax_fm_fake = self.discrim_enc(mask_recon_s)

                            with tf.variable_scope('Ah'):
                                Ah_dis_real, Ah_fm_real = self.discrim_classify(Ax_feats_real)
                                Ah_dis_fake, Ah_fm_fake = self.discrim_classify(Ax_feats_fake)

                            discrim_out_mask.append((Ah_dis_real, Ah_dis_fake))

                            discrim_fm_real_mask += Ax_fm_real + Ah_fm_real
                            discrim_fm_fake_mask += Ax_fm_fake + Ah_fm_fake

            vizN('A_recon', [mask, gen_mask])

        def LSGAN_hinge_loss(real, fake):
            d_real = tf.reduce_mean(-tf.minimum(0., tf.subtract(real, 1.)), name='d_real')
            d_fake = tf.reduce_mean(-tf.minimum(0., tf.add(-fake,-1.)), name='d_fake')
            d_loss = tf.multiply(d_real + d_fake, 0.5, name='d_loss')

            g_loss = tf.reduce_mean(-fake, name='g_loss')
            # add_moving_summary(g_loss)
            return g_loss, d_loss

        with tf.name_scope('losses'):
            with tf.name_scope('mask_losses'):
                with tf.name_scope('GAN_loss'):
                    # gan loss
                    G_loss_mask, D_loss_mask = zip(*[LSGAN_hinge_loss(real, fake) for real, fake in discrim_out_mask])
                    G_loss_mask = tf.add_n(G_loss_mask, name='mask_lsgan_loss')
                    D_loss_mask = tf.add_n(D_loss_mask, name='mask_Disc_loss')
                with tf.name_scope('FM_loss'):
                    FM_loss_mask = self.get_feature_match_loss(discrim_fm_real_mask, discrim_fm_fake_mask, 'mask_fm_loss')
                with tf.name_scope('z_recon_loss'):
                    z_recon_loss = tf.reduce_mean(tf.abs(z - z_recon), name='z_recon_loss')


        LAMBDA = 10.0
        self.g_loss = G_loss_mask/DIS_SCALE + LAMBDA*FM_loss_mask + LAMBDA*z_recon_loss
        self.d_loss = D_loss_mask
        self.collect_variables('gen', 'discrim')

        #add_moving_summary(G_loss_mask, D_loss_mask, FM_loss_mask)
        add_moving_summary(G_loss_mask, D_loss_mask,  FM_loss_mask, z_recon_loss)

    def optimizer(self):
        lr = tf.get_variable('learning_rate', initializer=2e-4, trainable=False)
        return tf.train.AdamOptimizer(lr, beta1=0.5, epsilon=1e-3)


def export_compact(model_path):
    """Export trained model to use it as a frozen and pruned inference graph in
       mobile applications. """
    pred_config = PredictConfig(
        session_init=SmartInit(model_path),
        model=Model(),
        input_names=['input', 'template', 'bbx', 'z1'],
        output_names=['gen/genmask/convlast/output'])
    ModelExporter(pred_config).export_compact(os.path.join(os.path.dirname(model_path), 'frozen_model.pb'))
    # ModelExporter(pred_config).export_compact(model_path)

def get_data(dataset, datadir, isTrain=True):

    def get_images(dir1, image_path):
        def get_df(dir):
            files = sorted(glob.glob(os.path.join(dir, '*.png')))
            df = CocoLoader(files, image_path, SHAPE, STYLE_DIM, STYLE_DIM_z2, channel=3, shuffle=isTrain)
            return df
        return get_df(dir1)

    # names = ['trainA', 'trainB'] if isTrain else ['testA', 'testB']
    path_type = 'train' if isTrain else 'val'
    path = os.path.join(args.path, '%s2017' % path_type)
    npy_path = os.path.join(args.data, path_type, dataset)
    df = get_images(npy_path, path)
    df = BatchData(df, BATCH if isTrain else TEST_BATCH)
    df = PrefetchDataZMQ(df, 2 if isTrain else 1)
    return df


class VisualizeTestSet(Callback):
    def _setup_graph(self):
        self.pred = self.trainer.get_predictor(
            ['input', 'template',  'mask', 'bbx', 'z1', 'z2'], ['A_recon/viz'])
    def _before_train(self):
        global args
        self.val_ds = get_data(args.dataset, args.data, isTrain=False)
        self.val_ds.reset_state()

    def _trigger(self):
        idx = 0
        for iA, tA, mA, bA, z1, z2 in self.val_ds:
            vizA = self.pred(iA, tA, mA, bA, z1, z2)
            self.trainer.monitors.put_image('testA-{}'.format(idx), vizA[0])
            # self.trainer.monitors.put_image('testB-{}'.format(idx), vizB)
            idx += 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dataset', required=True,
        help='name of the class used')
    parser.add_argument(
        '--data', required=True,
        help='directory containing bounding box annotations, should contain train, val folders')
    parser.add_argument(
        '--path', default='/data/jhtlab/deep/datasets/coco/',
        help='the path that contains the raw coco JPEG images')
    parser.add_argument('--nb_gpu', default='0', help='nb gpus to use, to use four gpus specify 0,1,2,3')
    parser.add_argument('--load', help='load model')
    args = parser.parse_args()

    # os.environ["CUDA_VISIBLE_DEVICES"] = args.nb_gpu
    nr_tower = max(get_num_gpu(), 1)
    # BATCH = BATCH * nr_tower
    mod = sys.modules['__main__']
    basename = os.path.basename(mod.__file__).split('.')[0]
    logdir = os.path.join('train_log', args.dataset, basename, datetime.now().strftime("%Y%m%d-%H%M%S"))
    logger.set_logger_dir(logdir)

    df = get_data(args.dataset, args.data)
    df = PrintData(df)
    data = QueueInput(df)

    GANTrainer(data, Model(), num_gpu=nr_tower).train_with_defaults(
        callbacks=[
            CheckNumerics(),
            PeriodicTrigger(ModelSaver(), every_k_epochs=500),
            ScheduledHyperParamSetter(
                'learning_rate',
                [(500, 2e-4), (1000, 0)], interp='linear'),
            PeriodicTrigger(VisualizeTestSet(), every_k_epochs=5),
        ],
        max_epoch=1000,
        steps_per_epoch=data.size() // nr_tower,
        session_init=SaverRestore(args.load) if args.load else None
    )

    export_compact(os.path.join(logdir, 'checkpoint'))