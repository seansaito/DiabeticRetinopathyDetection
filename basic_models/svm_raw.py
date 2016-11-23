import scipy.misc as sm
import numpy as np
#sm.imread("./sample/10_left.jpeg")
sm.imread("./sample/10_left.jpeg", mode="L")


import glob
import re
import random
import pandas as pd
from skimage.transform import resize



files = glob.glob("/media/3d352c20-c87e-44a3-8c27-e48382c61c7b/kaggle/train/*.jpeg")






def extractSampleLabel(filename):
	return re.search("\/([^\/]+)\.jpeg$", filename).group(1)



def readLabelsFile(labelFile):
	labels = []
	with open(labelFile, 'r') as f:
		for line in f:
			labels.append(line.strip().split(','))
	labels = labels[1:len(labels)] # Remove header line
	return labels


def readImageGreyscale(filename):
	return sm.imread(filename, mode="L")



labelFile = '/media/3d352c20-c87e-44a3-8c27-e48382c61c7b/kaggle/trainLabels.csv'
labelsArr = readLabelsFile(labelFile)



df = pd.read_csv(labelFile)


def generateFilePath(folder, labels):
	filePaths = []
	for singleLabel in labels:
		fullPath = folder + singleLabel + '.jpeg'
		filePaths.append(fullPath)
	return filePaths

labels = list(df['image'])
filePaths = generateFilePath("/media/3d352c20-c87e-44a3-8c27-e48382c61c7b/kaggle/train/", labels)

df['filePaths'] = filePaths



'''
labelsArr = []
for singleFile in files:
	labelForFile = extractSampleLabel(singleFile)
	labelsArr.append(labelForFile)
'''


# Randomly select 10% of the images for testing
totalImages = len(df)
numberofImagesForTesting = int(0.1 * totalImages)
random.seed(12345)
testingImageIndex = random.sample(range(0,totalImages), numberofImagesForTesting)


# Get index for training dataset
trainingImageIndex = []
for i in range(0,totalImages):
	if not i in testingImageIndex:
		trainingImageIndex.append(i)




for i in range:
	readImageGreyscale()










from sklearn import svm
X = [[0, 0], [2, 2]]
y = [0.5, 2.5]
clf = svm.SVR()
clf.fit(X, y)
clf.predict([[1, 1]])





from sklearn import svm
X = np.array([[0, 0], [2, 2]])
y = [0.5, 2.5]
clf = svm.SVR()
clf.fit(X, y)
clf.predict([[1, 1]])






df.iloc[0:10, 2:3]


imageMultiArr = np.array
for imagefile in df['filePaths'][1:10]:
	imageArr = readImageGreyscale(imagefile)
	imageMultiArr.append(np.ndarray.flatten(imageArr))

df['level'][1:10]


np.ndarray.flatten(readImageGreyscale('/media/3d352c20-c87e-44a3-8c27-e48382c61c7b/kaggle/train/10_left.jpeg')).shape








def prepareImageForSVM(imageArr):
	imageArrResize = resize(imageArr, (512,768), mode='nearest')
	imageArrResizeFlatten = np.ndarray.flatten(imageArrResize).reshape((393216, 1))
	return imageArrResizeFlatten







# Extract the data for training from 10 images
imageMultiArr = np.empty((393216,0), dtype=int)
for imagefile in df['filePaths'][1:10]:
	imageArr = readImageGreyscale(imagefile)
	#imageMultiArr.append(np.ndarray.flatten(imageArr))
	imageArrResize = resize(imageArr, (512,768), mode='nearest')
	imageArrResizeFlatten = np.ndarray.flatten(imageArrResize).reshape((393216, 1))
	print "============"
	print imageArrResizeFlatten.shape
	print imageMultiArr.shape
	imageMultiArr = np.append(imageMultiArr, imageArrResizeFlatten, axis=1)






# Get a single image of Test Data
testImageArr = readImageGreyscale('/media/3d352c20-c87e-44a3-8c27-e48382c61c7b/kaggle/train/10_left.jpeg')
testImageArrPrep = prepareImageForSVM(testImageArr)



# Fit model and test on the test data
clf = svm.SVR()
clf.fit(imageMultiArr.T, list(df['level'][1:10]))
clf.predict(testImageArrPrep.T)
