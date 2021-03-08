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
from GAN_bicycle import GANTrainer, GANModelDesc
import os
import sys
import six
from datetime import datetime


DIS_SCALE = 3
SHAPE = 256
BATCH = 2
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
slim = tf.contrib.slim


def downsample(img):
    return tf.layers.average_pooling2d(img, 3, 2)

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

def down_sample_avg(x, scale_factor=2) :
    return tf.layers.average_pooling2d(x, pool_size=3, strides=scale_factor, padding='SAME')

def z_sample(mean, logvar):
    eps = tf.random_normal(tf.shape(mean), mean=0.0, stddev=1.0, dtype=tf.float32)

    return mean + tf.exp(logvar * 0.5) * eps

def kl_loss(mean, logvar):
    # shape : [batch_size, channel]
    loss = 0.5 * tf.reduce_sum(tf.square(mean) + tf.exp(logvar) - 1 - logvar)
    # loss = tf.reduce_mean(loss)
    return loss

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

def percep_loss(fs, loss_weights):
    flist=[]
    k=0
    for f in fs:
        f_1, f_2 = tf.split(f, 2, 0)
        flist.append(loss_weights[k]*tf.reduce_mean(tf.abs(f_1-f_2)))
        k+=1
    return tf.add_n(flist, name='ploss')

def GNLReLU(x, name=None):
    x = GroupNorm(x)
    return tf.nn.leaky_relu(x, name=name)

def INLReLU(x, name=None):
    x = InstanceNorm('in', x)
    return tf.nn.leaky_relu(x, name=name)

def noisyINLReLU(x, name=None):
    x = tf.keras.layers.GaussianNoise(1)(x, training=True)
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
        with tf.variable_scope(name), argscope([Conv2D]):
            input = x
            x = Conv2D('conv0', x, chan, 3, activation=noisyINLReLU, strides=1)
            x = Conv2D('conv1', x, chan, 3, activation=noisyINLReLU, strides=1)
            return x+input

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
    def vgg_16(self, inputs,
               num_classes=1000,
               is_training=False,
               dropout_keep_prob=0.5,
               spatial_squeeze=True,
               scope='vgg_16',
               fc_conv_padding='VALID',
               global_pool=False,
               reuse=False):
        """Oxford Net VGG 16-Layers version D Example.

        Note: All the fully_connected layers have been transformed to conv2d layers.
              To use in classification mode, resize input to 224x224.

        Args:
          inputs: a tensor of size [batch_size, height, width, channels].
          num_classes: number of predicted classes. If 0 or None, the logits layer is
            omitted and the input features to the logits layer are returned instead.
          is_training: whether or not the model is being trained.
          dropout_keep_prob: the probability that activations are kept in the dropout
            layers during training.
          spatial_squeeze: whether or not should squeeze the spatial dimensions of the
            outputs. Useful to remove unnecessary dimensions for classification.
          scope: Optional scope for the variables.
          fc_conv_padding: the type of padding to use for the fully connected layer
            that is implemented as a convolutional layer. Use 'SAME' padding if you
            are applying the network in a fully convolutional manner and want to
            get a prediction map downsampled by a factor of 32 as an output.
            Otherwise, the output prediction map will be (input / 32) - 6 in case of
            'VALID' padding.
          global_pool: Optional boolean flag. If True, the input to the classification
            layer is avgpooled to size 1x1, for any input size. (This is not part
            of the original VGG architecture.)

        Returns:
          net: the output of the logits layer (if num_classes is a non-zero integer),
            or the input to the logits layer (if num_classes is 0 or None).
          end_points: a dict of tensors with intermediate activations.
        """
        with tf.variable_scope(scope, 'vgg_16', [inputs]) as sc:
            out = []
            end_points_collection = sc.original_name_scope + '_end_points'
            # Collect outputs for conv2d, fully_connected and max_pool2d.
            with slim.arg_scope([slim.conv2d, slim.fully_connected, slim.max_pool2d],
                                outputs_collections=end_points_collection):
                net = slim.repeat(inputs, 2, slim.conv2d, 64, [3, 3], scope='conv1')

                # with tf.variable_scope('relu1'):
                out1 = net

                net = slim.max_pool2d(net, [2, 2], scope='pool1')
                net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope='conv2')

                # with tf.variable_scope('relu2'):
                # out = tf.add(net, tf.zeros_like(net), name='conv2_2')
                out2 = net

                net = slim.max_pool2d(net, [2, 2], scope='pool2')
                net = slim.repeat(net, 3, slim.conv2d, 256, [3, 3], scope='conv3')

                # with tf.variable_scope('relu3'):
                out3 = net

                net = slim.max_pool2d(net, [2, 2], scope='pool3')
                net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv4')

                out4 = net

                return out1, out2, out3, out4

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
        with tf.variable_scope('senc'), argscope([tf.layers.conv2d, Conv2D]):

            x = tpad(x, pad=1, mode='reflect')
            x = tf.layers.conv2d(x, chan, kernel_size=3, strides=2, activation=INLReLU, name='conv_0')

            for i in range(3):
                x = tpad(x, pad=1, mode='reflect')
                x = tf.layers.conv2d(x, chan*2,  kernel_size=3, strides=2, activation=INLReLU, name='conv_%d' % (i+1))
                chan*=2

            x = tf.layers.conv2d(x, chan, kernel_size=3, strides=2, activation=INLReLU, name='conv_%d' % (i + 2))
            x = tf.layers.conv2d(x, chan, kernel_size=3, strides=2, activation=INLReLU, name='conv_%d' % (i + 3))

            x = tf.layers.flatten(x)
            mean = tf.layers.dense(x, STYLE_DIM_z2, name='fcmean')
            mean = tf.reshape(mean, [-1, 1, 1, STYLE_DIM_z2])
            var = tf.layers.dense(x, STYLE_DIM_z2, name='fcvar')
            var = tf.reshape(var, [-1, 1, 1, STYLE_DIM_z2])
        return mean, var

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
    def generator(self, img, z, nb_blocks=9, chan=3):
        assert img is not None
        with argscope([Conv2D, Conv2DTranspose]):
            x = tf.concat([img, z], axis=-1)
            x = tf.pad(x, [[0, 0], [3, 3], [3, 3], [0, 0]], mode='SYMMETRIC')
            x = Conv2D('conv0', x, NF, 7, activation=tf.nn.leaky_relu, padding='VALID')
            x = tf.concat([x, z], axis=-1)
            x = Conv2D('conv1', x, NF*2, 3, strides=2, activation=INLReLU)
            x = tf.concat([x, down_sample_avg(z,2)], axis=-1)
            x = Conv2D('conv2', x, NF*4, 3, strides=2, activation=INLReLU)
            x = tf.concat([x, down_sample_avg(z,4)], axis=-1)
            x = Conv2D('conv3', x, NF*8, 3, strides=2, activation=INLReLU)
            x = tf.concat([x, down_sample_avg(z,8)], axis=-1)

            for k in range(nb_blocks):
                x = Model.build_res_block(x, 'res{}'.format(k), NF * 8 + STYLE_DIM_z2, first=(k == 0))

            x = tf.keras.layers.UpSampling2D(2)(x)
            x = Conv2D('conv4', x, NF * 4, 3, strides=1, activation=noisyINLReLU)
            x = tf.keras.layers.UpSampling2D(2)(x)
            x = Conv2D('conv5', x, NF*2, 3, strides=1, activation=noisyINLReLU)
            x = tf.keras.layers.UpSampling2D(2)(x)
            x = Conv2D('conv6', x, NF, 3, strides=1, activation=noisyINLReLU)
            x = tf.pad(x, [[0, 0], [3, 3], [3, 3], [0, 0]], mode='SYMMETRIC')
            x = Conv2D('convlast', x, chan, 7, strides=1, padding='VALID', activation=tf.tanh, use_bias=True)

        return x

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
            mask = (mask / 128.0 - 1.0)

        with tf.name_scope('styleIn'):
            style_shape_z2 = [tf.shape(mask)[0], 1, 1, STYLE_DIM_z2]
            z3 = tf.random_normal(style_shape_z2, mean=0.0, stddev=1.0, dtype=tf.float32, name='z3')

        def vizN(name, a):
            with tf.name_scope(name):
                im = tf.concat(a, axis=2)
                im = (im + 1.0) * 128
                im = tf.clip_by_value(im, 0, 255)
                im = tf.cast(im, tf.uint8, name='viz')
            tf.summary.image(name, im, max_outputs=50)

        # use the initializers from torch
        with argscope([Conv2D, Conv2DTranspose, tf.layers.conv2d]):
            #Let us encode the images
            with tf.variable_scope('gen'):

                bin_gen_mask_gt = tf.round((mask + 1) * 0.5)
                in_gen_gt = img *(1-bin_gen_mask_gt) - bin_gen_mask_gt

                with tf.variable_scope('senc'):
                    zgt_mean, zgt_var = self.style_encoder(img*bin_gen_mask_gt)

                zgt = z_sample(zgt_mean, zgt_var)
                zmat = tf.tile(zgt, [1, in_gen_gt.shape[1], in_gen_gt.shape[2], 1])
                z2mat = tf.tile(z2, [1, in_gen_gt.shape[1], in_gen_gt.shape[2], 1])
                z3mat = tf.tile(z3, [1, in_gen_gt.shape[1], in_gen_gt.shape[2], 1])
                with tf.variable_scope('genRGB'):
                    gen_im = self.generator(in_gen_gt, z2mat, nb_blocks)
                    gen_im = gen_im*bin_gen_mask_gt + img*(1 - bin_gen_mask_gt)

                    gen_im_z3 = self.generator(in_gen_gt, z3mat, nb_blocks)
                    gen_im_z3 = gen_im_z3*bin_gen_mask_gt + img*(1 - bin_gen_mask_gt)

                    gen_im_gt = self.generator(in_gen_gt, zmat, nb_blocks)
                    gen_im_gt = gen_im_gt*bin_gen_mask_gt + img*(1 - bin_gen_mask_gt)

                with tf.variable_scope('senc'):
                    z3_recon, _ = self.style_encoder(gen_im_z3*bin_gen_mask_gt)

            f1, f2, f3, f4 = self.vgg_16(tf.concat([(img+1)*0.5, (gen_im_gt+1)*0.5], axis=0))


            #The final discriminator that takes them both
            discrim_out_mask = []
            discrim_fm_real_mask = []
            discrim_fm_fake_mask = []

            discrim_out = []
            discrim_out_z3 = []
            discrim_fm_real = []
            discrim_fm_fake = []

            with tf.variable_scope('discrim'):

                with tf.variable_scope('discrim_im'):

                    D_input_real = tf.concat([img, mask], axis=-1)
                    D_input_fake = tf.concat([gen_im_gt, mask], axis=-1)
                    D_inputs = [D_input_real, D_input_fake]

                    for s in range(DIS_SCALE):
                        with tf.variable_scope('s%d'%s):
                            if s != 0:
                                D_inputs = [downsample(im) for im in D_inputs]

                            im_s, im_recon_s = D_inputs

                            with tf.variable_scope('Ax'):
                                Ax_feats_real, Ax_fm_real = self.discrim_enc(im_s)
                                Ax_feats_fake, Ax_fm_fake = self.discrim_enc(im_recon_s)

                            with tf.variable_scope('Ah'):
                                Ah_dis_real, Ah_fm_real = self.discrim_patch_classify(Ax_feats_real)
                                Ah_dis_fake, Ah_fm_fake = self.discrim_patch_classify(Ax_feats_fake)

                            discrim_out.append((Ah_dis_real, Ah_dis_fake))
                            discrim_fm_real += Ax_fm_real + Ah_fm_real
                            discrim_fm_fake += Ax_fm_fake + Ah_fm_fake

                with tf.variable_scope('discrim_im', reuse=True):

                    D_input_real = tf.concat([img, mask], axis=-1)
                    D_input_fake = tf.concat([gen_im_z3, mask], axis=-1)
                    D_inputs = [D_input_real, D_input_fake]

                    for s in range(DIS_SCALE):
                        with tf.variable_scope('s%d'%s):
                            if s != 0:
                                D_inputs = [downsample(im) for im in D_inputs]

                            im_s, im_recon_s = D_inputs

                            with tf.variable_scope('Ax'):
                                Ax_feats_real, _ = self.discrim_enc(im_s)
                                Ax_feats_fake, _ = self.discrim_enc(im_recon_s)

                            with tf.variable_scope('Ah'):
                                Ah_dis_real, _ = self.discrim_patch_classify(Ax_feats_real)
                                Ah_dis_fake, _ = self.discrim_patch_classify(Ax_feats_fake)

                            discrim_out_z3.append((Ah_dis_real, Ah_dis_fake))


            vizN('A_recon', [img, gen_im_gt, gen_im_z3, gen_im])

        def LSGAN_hinge_loss(real, fake):
            d_real = tf.reduce_mean(-tf.minimum(0., tf.subtract(real, 1.)), name='d_real')
            d_fake = tf.reduce_mean(-tf.minimum(0., tf.add(-fake,-1.)), name='d_fake')
            d_loss = tf.multiply(d_real + d_fake, 0.5, name='d_loss')

            g_loss = tf.reduce_mean(-fake, name='g_loss')
            # add_moving_summary(g_loss)
            return g_loss, d_loss

        numelmask = tf.reduce_sum(bin_gen_mask_gt, axis=[1, 2, 3])
        numelall = tf.ones_like(numelmask) * SHAPE * SHAPE
        numelmask = tf.where(tf.equal(numelmask, 0), numelall, numelmask)
        weight_recon_loss = numelall / numelmask
        with tf.name_scope('losses'):
            with tf.name_scope('RGB_losses'):
                with tf.name_scope('GAN_loss'):
                    # gan loss
                    G_loss, D_loss = zip(*[LSGAN_hinge_loss(real, fake) for real, fake in discrim_out])
                    G_loss = tf.add_n(G_loss, name='lsgan_loss')
                    D_loss = tf.add_n(D_loss, name='Disc_loss')
                with tf.name_scope('GAN_loss_z3'):
                    # gan loss
                    G_loss_z3, D_loss_z3 = zip(*[LSGAN_hinge_loss(real, fake) for real, fake in discrim_out_z3])
                    G_loss_z3 = tf.add_n(G_loss_z3, name='lsgan_loss')
                    D_loss_z3 = tf.add_n(D_loss_z3, name='Disc_loss')
                with tf.name_scope('z_recon_loss'):
                    z3_recon_loss = tf.reduce_mean(tf.abs(z3 - z3_recon), name='z3_recon_loss')
                with tf.name_scope('FM_loss'):
                    FM_loss = [tf.reduce_mean(tf.abs(j - k))for j,k in zip(discrim_fm_real, discrim_fm_fake)]
                    FM_loss = tf.add_n(FM_loss)/len(FM_loss)
                with tf.name_scope('im_recon_loss'):
                    im_recon_loss = tf.reduce_mean(tf.reduce_mean(tf.abs(img - gen_im_gt), axis=[1,2,3])*weight_recon_loss)
                with tf.name_scope('kl_loss'):
                    KLloss = kl_loss(zgt_mean, zgt_var)
                with tf.name_scope('perceptualLoss'):
                    f3_1, f3_2 = tf.split(f3, 2, 0)
                    # perceptual_loss = tf.reduce_mean(tf.reduce_mean(tf.squared_difference(f3_1, f3_2), axis=[1,2,3])*weight_recon_loss)
                    perceptual_loss = tf.nn.l2_loss(f3_1-f3_2)/tf.to_float(tf.size(f3_1))

                    # perceptual_loss = percep_loss([f1, f2, f3, f4], [1.0 / 16, 1.0 / 8, 1.0 / 4, 1.0])




        LAMBDA = 10.0
        LAMBDA_KL = 0.05
        self.g_loss = G_loss/DIS_SCALE + G_loss_z3/DIS_SCALE + LAMBDA*FM_loss + LAMBDA*im_recon_loss + LAMBDA_KL*KLloss \
                      + LAMBDA*perceptual_loss
        self.d_loss = D_loss + D_loss_z3
        self.z_loss = LAMBDA * z3_recon_loss
        self.collect_variables('gen', 'discrim')

        tf.summary.histogram('z_var', zgt_var)
        tf.summary.histogram('z_mean', zgt_mean)
        add_moving_summary(G_loss, D_loss, FM_loss, im_recon_loss,
                           KLloss, z3_recon_loss, perceptual_loss)


    def optimizer(self):
        lr = tf.get_variable('learning_rate', initializer=2e-4, trainable=False)
        return tf.train.AdamOptimizer(lr, beta1=0.5, epsilon=1e-3)


def export_compact(model_path):
    """Export trained model to use it as a frozen and pruned inference graph in
       mobile applications. """
    pred_config = PredictConfig(
        session_init=SmartInit(model_path),
        model=Model(),
        input_names=['input', 'mask', 'z2'],
        output_names=['gen/genRGB/add'])
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
    BATCH = BATCH * nr_tower
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
            PeriodicTrigger(ModelSaver(), every_k_epochs=200),
            ScheduledHyperParamSetter(
                'learning_rate',
                [(200, 7e-4), (400, 0)], interp='linear'),
            PeriodicTrigger(VisualizeTestSet(), every_k_epochs=3),
        ],
        max_epoch=400,
        steps_per_epoch=data.size() // nr_tower,
        session_init=SaverRestore(args.load) if args.load else None
    )

    export_compact(os.path.join(logdir, 'checkpoint'))
