from pycocotools.coco import COCO
from scipy import ndimage
from tqdm import tqdm
import numpy as np
import os
import pylab
import sys

pylab.rcParams['figure.figsize'] = (8.0, 10.0)
from PIL import Image, ImagePalette # For indexed images
import matplotlib # For Matlab's color maps
import argparse

def cocoSegmentationToSegmentationMap(coco, imgId, catId=None, checkUniquePixelLabel=True, includeCrowd=False):
    '''
    Convert COCO GT or results for a single image to a segmentation map.
    :param coco: an instance of the COCO API (ground-truth or result)
    :param imgId: the id of the COCO image
    :param checkUniquePixelLabel: (optional) whether every pixel can have at most one label
    :param includeCrowd: whether to include 'crowd' thing annotations as 'other' (or void)
    :return: labelMap - [h x w] segmentation map that indicates the label of each pixel
    '''

    # Init
    curImg = coco.imgs[imgId]
    imageSize = (curImg['height'], curImg['width'])
    labelMap = np.zeros(imageSize)

    # Get annotations of the current image (may be empty)
    imgAnnots = [a for a in coco.anns.values() if a['image_id'] == imgId]
    if includeCrowd:
        annIds = coco.getAnnIds(imgIds=imgId)
    else:
        annIds = coco.getAnnIds(imgIds=imgId, iscrowd=False)
    imgAnnots = coco.loadAnns(annIds)

    # Combine all annotations of this image in labelMap
    #labelMasks = mask.decode([a['segmentation'] for a in imgAnnots])
    boxes=[]
    segmentations=[]
    for a in range(0, len(imgAnnots)):
        newLabel = imgAnnots[a]['category_id']
        if catId != None and catId != newLabel:
            #print('Skipping')
            continue
        labelMask = coco.annToMask(imgAnnots[a]) == 1
        if np.sum(labelMask*1.) / np.size(labelMask) < 0.01:
            continue
        labelMap = np.zeros(imageSize)
        box = [int(c) for c in imgAnnots[a]['bbox']]  # .toBbox()#coco.annToMask(imgAnnots[a]) == 1
        box[2] += box[0]
        box[3] += box[1]
        boxes.append(np.expand_dims(box, axis=1))


        if checkUniquePixelLabel and (np.logical_and(labelMap[labelMask] != newLabel,
            labelMap[labelMask] != 0)).any():
            raise Exception('Error: Some pixels have more than one label (image %d)!' % (imgId))

        labelMap[labelMask] = newLabel
        segmentations.append(labelMap)

    return segmentations, boxes

def cocoSegmentationToPng(coco, imgId, pngPath, includeCrowd=False):
    '''
    Convert COCO GT or results for a single image to a segmentation map and write it to disk.
    :param coco: an instance of the COCO API (ground-truth or result)
    :param imgId: the COCO id of the image (last part of the file name)
    :param pngPath: the path of the .png file
    :param includeCrowd: whether to include 'crowd' thing annotations as 'other' (or void)
    :return: None
    '''

    # Create label map
    labelMaps, boxes = cocoSegmentationToSegmentationMap(coco, imgId, includeCrowd=includeCrowd)
    labelMap = [labelmap.astype(np.int8) for labelmap in labelMaps]

    # Get color map and convert to PIL's format
    cmap = getCMap()
    cmap = (cmap * 255).astype(int)
    padding = np.zeros((256-cmap.shape[0], 3), np.int8)
    cmap = np.vstack((cmap, padding))
    cmap = cmap.reshape((-1))
    assert len(cmap) == 768, 'Error: Color map must have exactly 256*3 elements!'

    # Write to png file
    png = Image.fromarray(labelMap).convert('P')
    png.putpalette(cmap)
    png.save(pngPath, format='PNG')

def cocoSegmentationToPngBinary(coco, imgId, npyPath, catId, includeCrowd=False):
    '''
    Convert COCO GT or results for a single image to a segmentation map and write it to disk.
    :param coco: an instance of the COCO API (ground-truth or result)
    :param imgId: the COCO id of the image (last part of the file name)
    :param pngPath: the path of the .png file
    :param includeCrowd: whether to include 'crowd' thing annotations as 'other' (or void)
    :return: None
    '''

    # Create label map
    labelMaps, boxes = cocoSegmentationToSegmentationMap(coco, imgId, catId=catId, includeCrowd=includeCrowd)
    labelMaps = [labelmap.astype(np.int8) for labelmap in labelMaps]

    binary = [(labelmap > 0).astype(np.int8) for labelmap in labelMaps] #(labelMap > 0).astype(np.int8) * 255#.astype(float)
    counter=0
    for j in range(len(binary)):
        bj = binary[j]
        labeled, nr_objects = ndimage.label(bj)
        if nr_objects>1:
            continue
        if np.sum(bj) < 0.01*np.size(bj):
            continue
        # if np.sum(bj) > 0.9*np.size(bj):
        #     continue

        edg1 = np.sum(bj, axis=0)[:5].max()
        edg2 = np.sum(bj, axis=0)[-5:].max()
        edg3 = np.sum(bj, axis=1)[:5].max()
        edg4 = np.sum(bj, axis=1)[-5:].max()

        if edg1>0.01*np.size(bj, axis=0):
            continue
        if edg2>0.01*np.size(bj, axis=0):
            continue
        if edg3>0.01*np.size(bj, axis=1):
            continue
        if edg4>0.01*np.size(bj, axis=1):
            continue

        counter+=1
        bj = (bj*255).astype(np.uint8)
        Image.fromarray(bj).save(npyPath+'_%d.png'%counter)

def getCMap(stuffStartId=92, stuffEndId=182, cmapName='jet', addThings=True, addUnlabeled=True, addOther=True):
    '''
    Create a color map for the classes in the COCO Stuff Segmentation Challenge.
    :param stuffStartId: (optional) index where stuff classes start
    :param stuffEndId: (optional) index where stuff classes end
    :param cmapName: (optional) Matlab's name of the color map
    :param addThings: (optional) whether to add a color for the 91 thing classes
    :param addUnlabeled: (optional) whether to add a color for the 'unlabeled' class
    :param addOther: (optional) whether to add a color for the 'other' class
    :return: cmap - [c, 3] a color map for c colors where the columns indicate the RGB values
    '''

    # Get jet color map from Matlab
    labelCount = stuffEndId - stuffStartId + 1
    cmapGen = matplotlib.cm.get_cmap(cmapName, labelCount)
    cmap = cmapGen(np.arange(labelCount))
    cmap = cmap[:, 0:3]

    # Reduce value/brightness of stuff colors (easier in HSV format)
    cmap = cmap.reshape((-1, 1, 3))
    hsv = matplotlib.colors.rgb_to_hsv(cmap)
    hsv[:, 0, 2] = hsv[:, 0, 2] * 0.7
    cmap = matplotlib.colors.hsv_to_rgb(hsv)
    cmap = cmap.reshape((-1, 3))

    # Permute entries to avoid classes with similar name having similar colors
    st0 = np.random.get_state()
    np.random.seed(42)
    perm = np.random.permutation(labelCount)
    np.random.set_state(st0)
    cmap = cmap[perm, :]

    # Add black (or any other) color for each thing class
    if addThings:
        thingsPadding = np.zeros((stuffStartId - 1, 3))
        cmap = np.vstack((thingsPadding, cmap))

    # Add black color for 'unlabeled' class
    if addUnlabeled:
        cmap = np.vstack(((0.0, 0.0, 0.0), cmap))

    # Add yellow/orange color for 'other' class
    if addOther:
        cmap = np.vstack((cmap, (1.0, 0.843, 0.0)))

    return cmap


parser = argparse.ArgumentParser()
parser.add_argument(
    '--dataset', required=True,
    help='name of the class used')
parser.add_argument(
    '--path', required=True,
    help='directory containing COCO dataset')
parser.add_argument(
    '--mode', required=True,
    help='train or val')
args = parser.parse_args()

dataDir = args.path
dataType = '%s2017' % args.mode
annFile='{}/annotations/instances_{}.json'.format(dataDir,dataType)

coco=COCO(annFile)

cats = coco.loadCats(coco.getCatIds())
nms=[cat['name'] for cat in cats]
print('COCO categories: \n{}\n'.format(' '.join(nms)))

nms = set([cat['supercategory'] for cat in cats])
print('COCO supercategories: \n{}'.format(' '.join(nms)))

cat_names = [args.dataset]
catIds = coco.getCatIds(catNms=cat_names)
out_dir = 'dataset'
out_dir = os.path.join(out_dir, args.mode)
if not os.path.exists(out_dir):
    print('Creating %s' % out_dir)
    os.makedirs(out_dir)

for catId, name in zip(catIds, cat_names):
    imgIds = coco.getImgIds(catIds=catId)
    print('Extracting category %s' % name)
    cat_filepath = os.path.join(out_dir, name)
    if not os.path.exists(cat_filepath):
        os.mkdir(cat_filepath)
    print('Saving to directory %s' % cat_filepath)
    for imgId in tqdm(imgIds):
        outfile = os.path.join(cat_filepath, '%012d' % imgId)
        cocoSegmentationToPngBinary(coco, imgId, outfile, catId)