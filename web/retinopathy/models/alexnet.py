import caffe
import numpy as np
from PIL import Image

MODEL_FILE = "data/deploy_alexnet.prototxt"
WEIGHTS_FILE = "data/alexnet_iter_2000.caffemodel"
INPUT_SHAPE = (512, 512)


class AlexNet(object):
    def __init__(self):
        self.network = caffe.Net(MODEL_FILE, WEIGHTS_FILE, caffe.TEST)

    def predict(self, img):
        sample_images = []
        resized = img.resize(INPUT_SHAPE, Image.ANTIALIAS)
        nparr = np.array(resized)
        nparr = np.swapaxes(nparr, 0, 2)
        sample_images.append(nparr)

        # Feed data into network
        im_input = np.array(sample_images)
        caffe_network.blobs["data"].reshape(*im_input.shape)
        caffe_network.blobs["data"].data[...] = im_input

        # Inference
        result = self.network.forward()
        labels = np.argmax(result["loss"], axis=1)
        return labels[0]


