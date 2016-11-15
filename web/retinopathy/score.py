import caffe
import numpy as np
from PIL import Image

MODEL_FILE = "models/deploy_alexnet.prototxt"
WEIGHTS_FILE = "models/alexnet_iter_2000.caffemodel"
INPUT_SHAPE = (512, 512)

# Sets up the network
caffe_network = caffe.Net(MODEL_FILE, WEIGHTS_FILE, caffe.TEST)


def compute_score(img):
    # Resize and
    sample_images = []
    res = img.resize(INPUT_SHAPE, Image.ANTIALIAS)
    img = np.array(res)
    img = np.swapaxes(img, 0, 2)
    sample_images.append(img)

    # Feed data into network
    im_input = np.array(sample_images)
    caffe_network.blobs["data"].reshape(*im_input.shape)
    caffe_network.blobs["data"].data[...] = im_input

    # Inference
    res = caffe_network.forward()
    labels = np.argmax(res["loss"], axis=1)
    return labels[0]
