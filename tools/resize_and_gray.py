import scipy.misc as sm
import numpy as np
from skimage.transform import resize
import sys



def readImageGreyscale(filename):
	return sm.imread(filename, mode="L")


def resizeImage(imageArr):
	#imageArrResize = resize(imageArr, (512,768), mode='nearest')
	imageArrResize = resize(imageArr, (256,384), mode='nearest')
	#imageArrResizeFlatten = np.ndarray.flatten(imageArrResize).reshape((393216, 1))
	return imageArrResize


imageFile = sys.argv[1]
folder 	  = sys.argv[2]

filenameArr = imageFile.split('/')
filelabel = filenameArr[len(filenameArr) - 1]
resultFile = folder + filelabel

greyImage = readImageGreyscale(imageFile)
greyImageResize = resizeImage(greyImage)
sm.imsave(resultFile, greyImageResize)


#################################################
#################################################
