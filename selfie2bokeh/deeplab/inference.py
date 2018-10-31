import os
import tarfile
import tempfile
import urllib

from matplotlib import gridspec
from matplotlib import pyplot as plt
import numpy as np
from PIL import Image
import cv2

import tensorflow as tf


class DeepLabModel(object):
    """Class to load deeplab model and run inference."""

    INPUT_TENSOR_NAME = 'ImageTensor:0'
    OUTPUT_TENSOR_NAME = 'SemanticPredictions:0'
    FROZEN_GRAPH_NAME = 'frozen_inference_graph'

    def __init__(self, tarball_path, input_size=256):
        """Creates and loads pretrained deeplab model."""
        self.graph = tf.Graph()
        self.INPUT_SIZE = input_size
        graph_def = None
        # Extract frozen graph from tar archive.
        tar_file = tarfile.open(tarball_path)
        for tar_info in tar_file.getmembers():
            if self.FROZEN_GRAPH_NAME in os.path.basename(tar_info.name):
                file_handle = tar_file.extractfile(tar_info)
                graph_def = tf.GraphDef.FromString(file_handle.read())
                break

        tar_file.close()

        if graph_def is None:
            raise RuntimeError('Cannot find inference graph in tar archive.')

        with self.graph.as_default():
            tf.import_graph_def(graph_def, name='')

        self.sess = tf.Session(graph=self.graph)

    def run(self, image):
        """Runs inference on a single image.

        Args:
          image: A PIL.Image object, raw input image.

        Returns:
          resized_image: RGB image resized from original input image.
          seg_map: Segmentation map of `resized_image`.
        """
        width, height = image.size
        resize_ratio = 1.0 * self.INPUT_SIZE / max(width, height)
        target_size = (int(resize_ratio * width), int(resize_ratio * height))
        resized_image = image.convert('RGB').resize(
            target_size, Image.ANTIALIAS)
        batch_seg_map = self.sess.run(
            self.OUTPUT_TENSOR_NAME,
            feed_dict={self.INPUT_TENSOR_NAME: [np.asarray(resized_image)]})
        seg_map = batch_seg_map[0]
        return resized_image, seg_map


def create_pascal_label_colormap():
    """Creates a label colormap used in PASCAL VOC segmentation benchmark.

    Returns:
      A Colormap for visualizing segmentation results.
    """
    colormap = np.zeros((256, 3), dtype=int)
    ind = np.arange(256, dtype=int)

    for shift in reversed(range(8)):
        for channel in range(3):
            colormap[:, channel] |= ((ind >> channel) & 1) << shift
        ind >>= 3

    return colormap


def label_to_color_image(label):
    """Adds color defined by the dataset colormap to the label.

    Args:
      label: A 2D array with integer type, storing the segmentation label.

    Returns:
      result: A 2D array with floating type. The element of the array
        is the color indexed by the corresponding element in the input label
        to the PASCAL color map.

    Raises:
      ValueError: If label is not of rank 2 or its value is larger than color
        map maximum entry.
    """
    if label.ndim != 2:
        raise ValueError('Expect 2-D input label')

    colormap = create_pascal_label_colormap()

    if np.max(label) >= len(colormap):
        raise ValueError('label value too large.')

    return colormap[label]


def vis_segmentation(image, seg_map, save=False):
    """Visualizes input image, segmentation map and overlay view."""
    figure = plt.figure(figsize=(15, 5))
    grid_spec = gridspec.GridSpec(1, 4, width_ratios=[6, 6, 6, 6])

    # plot original image
    plt.subplot(grid_spec[0])
    plt.imshow(image)
    plt.axis('off')
    plt.title('input image')

    # plot segmentation mask
    plt.subplot(grid_spec[1])
    seg_map[seg_map != 15] = 0
    seg_image = label_to_color_image(seg_map).astype(np.uint8)
    plt.imshow(seg_image)
    plt.axis('off')
    plt.title('segmentation map')

    # mask out pixels from original image
    image_masked = image.convert("RGBA")
    pixdata = image_masked.load()
    width, height = image.size
    print(image.size)
    for y in range(height):
        for x in range(width):
            if seg_map[y, x] != 15:
                pixdata[x, y] = (255, 255, 255, 0)

    # create opencv image
    image_cv = np.array(image)

    # average blur
    kernel = np.ones((5, 5), np.float32) / 25
    image_blurred_average = cv2.filter2D(image_cv, -1, kernel)
    plt.subplot(grid_spec[2])
    plt.imshow(image_blurred_average)
    plt.imshow(image_masked)
    plt.axis('off')
    plt.title('average blur')

    # gaussian blur
    image_blurred_gaussian = cv2.GaussianBlur(image_cv, (9, 9), 0)
    plt.subplot(grid_spec[3])
    plt.imshow(image_blurred_gaussian)
    plt.imshow(image_masked)
    plt.axis('off')
    plt.title('gaussian blur')

    if save:
        return figure
    else:
        plt.show()


def run_visualization(model, image, save=False):
    """Inferences DeepLab model and visualizes result."""
    print('running deeplab on image %s...' % image)
    orignal_im = Image.open(image)
    resized_im, seg_map = model.run(orignal_im)

    if save:
        figure = vis_segmentation(resized_im, seg_map, save=save)
        return figure
    else:
        vis_segmentation(resized_im, seg_map)


def create_segmentation_model(input_size=256):
    MODEL_NAME = 'xception_coco_voctrainval'

    _DOWNLOAD_URL_PREFIX = 'http://download.tensorflow.org/models/'
    _MODEL_URLS = {
        'mobilenetv2_coco_voctrainaug':
            'deeplabv3_mnv2_pascal_train_aug_2018_01_29.tar.gz',
        'mobilenetv2_coco_voctrainval':
            'deeplabv3_mnv2_pascal_trainval_2018_01_29.tar.gz',
        'xception_coco_voctrainaug':
            'deeplabv3_pascal_train_aug_2018_01_04.tar.gz',
        'xception_coco_voctrainval':
            'deeplabv3_pascal_trainval_2018_01_04.tar.gz',
    }
    _TARBALL_NAME = 'deeplab_model.tar.gz'

    model_dir = tempfile.mkdtemp()
    tf.gfile.MakeDirs(model_dir)

    download_path = os.path.join(MODEL_NAME, _TARBALL_NAME)

    if not os.path.exists(download_path):
        os.makedirs(MODEL_NAME)
        print('downloading model, this might take a while...')
        urllib.request.urlretrieve(_DOWNLOAD_URL_PREFIX + _MODEL_URLS[MODEL_NAME],
                                   download_path)
        print('download completed! loading DeepLab model...')

    return DeepLabModel(download_path, input_size)


def create_segmentation_map(model, image):
    if type(image) is str:
        image = Image.open(image)
    resized_im, seg_map = model.run(image)

    image_masked = resized_im.convert("RGBA")
    pixdata = image_masked.load()
    width, height = resized_im.size
    for y in range(height):
        for x in range(width):
            if seg_map[y, x] != 15:
                pixdata[x, y] = (0, 0, 0, 0)

    return image_masked


def create_map_dataset(model, inputFolder, outputFolder):
    images = os.listdir(inputFolder)
    for image in images:
        mask = create_segmentation_map(model, os.path.join(inputFolder, image))
        mask.save(os.path.join(outputFolder, image), "png")


def create_masks():
    model = create_segmentation_model()
    inputFolder = "../datasets/selfie2bokeh/testA"
    outputFolder = "../datasets/selfie2bokeh/maskA"
    create_map_dataset(model, inputFolder, outputFolder)

    inputFolder = "../datasets/selfie2bokeh/testB"
    outputFolder = "../datasets/selfie2bokeh/maskB"
    create_map_dataset(model, inputFolder, outputFolder)


def run_tests(inputFolder, outputFolder):
    images = os.listdir(inputFolder)
    model = create_segmentation_model()

    for image in images:
        result = run_visualization(model, os.path.join(inputFolder, image), save=True)
        result.savefig(os.path.join(outputFolder, image))


run_tests(r'Z:\CV-Project\pytorch-CycleGAN-and-pix2pix\datasets\selfie2bokeh\testA', r'Z:\CV-Project\pytorch-CycleGAN-and-pix2pix\results\selfie2bokeh_conventional')
