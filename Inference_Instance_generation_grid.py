import argparse
import numpy as np
import tensorflow as tf
import cv2
from PIL import Image
import os
from dataloader import insertion_inference
import glob
SHAPE = 256
STYLE_DIM = 128
STYLE_DIM_z2 = 8

np.random.seed(0)
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
if __name__ == '__main__':
    # Let's allow the user to pass the filename as an argument
    parser = argparse.ArgumentParser()
    parser.add_argument("--mask_model", default="results/frozen_model.pb", type=str,
                        help="Frozen model file to import")
    parser.add_argument("--rgb_model", default="results/frozen_model.pb", type=str,
                        help="Frozen model file to import")
    parser.add_argument("--mask_folder", required=True, type=str, help="Folder containing masks")
    parser.add_argument("--bg_folder", required=True, type=str, help="Frozen containing background images")
    args = parser.parse_args()

    # We use our "load_graph" function
    # We load the protobuf file from the disk and parse it to retrieve the
    # unserialized graph_def
    with tf.gfile.GFile(args.mask_model, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
    with tf.Graph().as_default() as graph_mask:
        tf.import_graph_def(graph_def, name='')

    # We access the input and output nodes
    im = graph_mask.get_tensor_by_name('input:0')
    t = graph_mask.get_tensor_by_name('template:0')
    bbx = graph_mask.get_tensor_by_name('bbx:0')
    z = graph_mask.get_tensor_by_name('z1:0')
    genm = graph_mask.get_tensor_by_name('gen/genmask/convlast/output:0')


    with tf.gfile.GFile(args.rgb_model, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
    with tf.Graph().as_default() as graph_rgb:
        tf.import_graph_def(graph_def, name='')

    im_rgb = graph_rgb.get_tensor_by_name('input:0')
    mask = graph_rgb.get_tensor_by_name('mask:0')
    z2 = graph_rgb.get_tensor_by_name('z2:0')
    genim = graph_rgb.get_tensor_by_name('gen/genRGB/add:0')

    m_files = glob.glob(os.path.join(args.mask_folder, '*.png'))[:10] #Remove indexing to use all the validation masks
    im_files = glob.glob(os.path.join(args.bg_folder, '*.jpg'))

    dataloader = insertion_inference(m_files, im_files, SHAPE, STYLE_DIM, STYLE_DIM_z2)
    savepath = os.path.join(os.path.dirname(args.rgb_model), 'results')
    os.makedirs(savepath, exist_ok=True)
    for its, (np_im, template, np_m, r, z_np, z2_np) in enumerate(dataloader):
        # grid_im = Image.new('RGB', (SHAPE*10, SHAPE*10))
        sep_v = 3*3
        sep_h = 3*3
        grid_m = Image.new('L', (SHAPE * 3 + sep_h, SHAPE * 3 + sep_v))
        grid_im = Image.new('RGBA', (SHAPE * 3 + sep_h, SHAPE * 3 +sep_v), (255, 255, 255, 255))
        counteri = 0
        m_finals = []
        with tf.Session(graph=graph_mask) as sess_mask:
            for i in range(0, SHAPE * 3, SHAPE):
                counterj = 0
                for j in range(0, SHAPE * 3, SHAPE):

                    z_np = np.random.normal(size=[1, 1, 1, STYLE_DIM])
                    m_final = sess_mask.run(genm, feed_dict={im: np_im,
                                                            t: template,
                                                            z: z_np,
                                                            bbx:r})
                    m_finals.append((m_final+1)*0.5*255)
                    m_final = (np.squeeze(m_final) * 255).astype(np.uint8)
                    grid_m.paste(Image.fromarray(m_final), (i + counteri, j + counterj))
                    counterj+=3
                counteri+=3
            grid_m.save(os.path.join(savepath, 'm_grid_%d.png' % (its)))

        counteri = 0
        idx = 0
        with tf.Session(graph=graph_rgb) as sess_rgb:
            for i in range(0, SHAPE * 3, SHAPE):
                counterj = 0
                for j in range(0, SHAPE * 3, SHAPE):
                    z2_np = np.random.normal(size=[1, 1, 1, STYLE_DIM_z2])
                    im_final = sess_rgb.run(genim,
                                        feed_dict={im_rgb: np_im,
                                                   mask: m_finals[idx],
                                                   z2: z2_np})

                    im_final = (im_final + 1) * 0.5 * 255
                    im_final = im_final.squeeze().astype(np.uint8)
                    grid_im.paste(Image.fromarray(im_final), (i + counteri, j + counterj))
                    counterj+=3
                    idx+=1
                counteri+=3
            grid_im.save(os.path.join(savepath, 'im_grid_%d.png'%(its)))
        # its +=1
