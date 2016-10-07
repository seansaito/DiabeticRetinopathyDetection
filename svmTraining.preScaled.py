import scipy.misc as sm
import numpy as np
import glob
import re
import random
import pandas as pd
from skimage.transform import resize 
from sklearn import svm

#FOLDER = "/media/3d352c20-c87e-44a3-8c27-e48382c61c7b/kaggle/train/"
FOLDER = "/speed/kartong/kaggle/retinopathy/256_greyData/"
LABELFILE = "/speed/kartong/kaggle/trainLabels.csv"
#LABELFILE = "/media/3d352c20-c87e-44a3-8c27-e48382c61c7b/kaggle/trainLabels.csv"



def readLabelsFile(labelFile):
	labels = []
	with open(labelFile, 'r') as f:
		for line in f:
			labels.append(line.strip().split(','))  
	labels = labels[1:len(labels)] # Remove header line
	return labels



def readImageGreyscale(filename):
	return sm.imread(filename, mode="L")


def generateFilePath(folder, labels):
	'''
	Generate full filepaths for the image from its folder
	and labels
	'''
	filePaths = []
	for singleLabel in labels:
		fullPath = folder + singleLabel + '.jpeg'
		filePaths.append(fullPath)
	return filePaths



# Read Label file
labelFile = LABELFILE
labelsArr = readLabelsFile(labelFile)	
df = pd.read_csv(labelFile)


# Get filepaths from file labels and folder path
labels = list(df['image'])
filePaths = generateFilePath(FOLDER, labels)


# Add filepath column to dataframe
df['filePaths'] = filePaths


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


def prepareImageForSVM(imageArr):
	imageArrResizeFlatten = np.ndarray.flatten(imageArr).reshape((98304, 1))
	return imageArrResizeFlatten





#################################################
#################################################


# Extract the data for training from 90% of the data
imageMultiArr = np.empty((98304,0), dtype=int)
for i in trainingImageIndex[1:200]:
	print "Reading Image %s" % i
	imagefile = df['filePaths'][i]
	imageArr = readImageGreyscale(imagefile)
	imageArrFlatten = np.ndarray.flatten(imageArr).reshape((98304, 1))
	imageMultiArr = np.append(imageMultiArr, imageArrFlatten, axis=1)

clf = svm.SVR()
clf.fit(imageMultiArr.T, list(df['level'][trainingImageIndex[1:1200]]))




f = open("testResult.txt", "w")

# Do test on the testing images
for i in testingImageIndex[1:100]:
	print "Testing Image %s" %i

	# Prepare the testing image
	imagefile = df['filePaths'][i]
	testImageArr = readImageGreyscale(imagefile)
	testImageArrPrep = prepareImageForSVM(testImageArr)
	
	# Prediction with the model
	predictedVal = clf.predict(testImageArrPrep.T)
	print predictedVal

	# Write result to file
	resultLine = str(i) + "\t" + str(df['image'][i]) + "\t" + str(df['level'][i]) + "\t" + str(predictedVal[0]) + "\n"
	f.write(resultLine)


f.close()



