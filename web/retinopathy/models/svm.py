import os
import numpy as np
import scipy.misc as sm
import skimage.transform as skt
import sklearn.externals as ske


__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))

PIC_WIDTH = 384
PIC_HEIGHT = PIC_WIDTH / 3 * 2

class SvmBase(object):
    def __init__(self):
        self.clf = ske.joblib.load(self._model_file())

    def predict(self, img):
        gray = sm.fromimage(img, mode="L")
        resized = skt.resize(gray, (PIC_HEIGHT, PIC_WIDTH), mode="nearest")
        flattened = np.ndarray.flatten(resized).reshape((PIC_WIDTH * PIC_HEIGHT, 1))
        predicted_val = self.clf.predict(flattened.T)
        return int(round(predicted_val[0]))

    def _model_file(self):
        raise NotImplementedError("Please define a model file for this SVM class")


class SvmLinear2000(SvmBase):
    def _model_file(self):
        return os.path.join(__location__, "data/svmLinear.2000.pickle")


class SvmLinear5000(SvmBase):
    def _model_file(self):
        return os.path.join(__location__, "data/svmLinear.5000.pickle")


class SvmPoly2000(SvmBase):
    def _model_file(self):
        return os.path.join(__location__, "data/svmPoly.2000.pickle")


class SvmPoly5000(SvmBase):
    def _model_file(self):
        return os.path.join(__location__, "data/svmPoly.5000.pickle")

