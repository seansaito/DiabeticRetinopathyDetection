import scipy.misc as sm
import numpy as np
import re
import random
import pandas as pd
from sklearn.linear_model import LogisticRegression
import sys

#FOLDER = "/media/3d352c20-c87e-44a3-8c27-e48382c61c7b/kaggle/train/"
FOLDER = "/speed/kartong/kaggle/retinopathy/256_greyData/"
LABELFILE = "/speed/kartong/kaggle/trainLabels.csv"
#LABELFILE = "/media/3d352c20-c87e-44a3-8c27-e48382c61c7b/kaggle/trainLabels.csv"


TRAINING_SIZE = int(sys.argv[1])
PREDICTION_FILE = sys.argv[2]


# Size of image to resize and use for training
PIC_WIDTH = 384
PIC_HEIGHT = PIC_WIDTH / 3 * 2
TOTAL_PIXELS = PIC_WIDTH * PIC_HEIGHT



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
random.seed(1234567)
testingImageIndex = random.sample(range(0,totalImages), numberofImagesForTesting)


# Get index for training dataset
trainingImageIndex = []
for i in range(0,totalImages):
	if not i in testingImageIndex:
		trainingImageIndex.append(i)


def prepareImageForSVM(imageArr):
	imageArrResizeFlatten = np.ndarray.flatten(imageArr).reshape((TOTAL_PIXELS, 1))
	return imageArrResizeFlatten





#################################################
#################################################

# Extract the data for training from 90% of the data
numTraining = TRAINING_SIZE
imageMultiArr = np.empty((TOTAL_PIXELS,0), dtype=int)
for i in trainingImageIndex[1:numTraining]:
	print "Reading Image %s" % i
	imagefile = df['filePaths'][i]
	imageArr = readImageGreyscale(imagefile)
	imageArrFlatten = np.ndarray.flatten(imageArr).reshape((TOTAL_PIXELS, 1))
	imageMultiArr = np.append(imageMultiArr, imageArrFlatten, axis=1)


lr = LogisticRegression(solver='newton-cg', multi_class='multinomial', n_jobs=100)
lr.fit(imageMultiArr.T, list(df['level'][trainingImageIndex[1:numTraining]]))



f = open(PREDICTION_FILE, "w")

# Do test on the testing images
for i in testingImageIndex[0:200]:
#for i in testingImageIndex:
#for i in range(1,200):
	print "Testing Image %s" %i

	# Prepare the testing image
	imagefile = df['filePaths'][i]
	print imagefile
	testImageArr = readImageGreyscale(imagefile)
	testImageArrFlatten = np.ndarray.flatten(testImageArr).reshape((TOTAL_PIXELS, 1))
	testImageArrPrep = testImageArrFlatten
	
	# Prediction with the model
	predictedVal = lr.predict(testImageArrPrep.T)
	print predictedVal

	# Write result to file
	resultLine = str(i) + "\t" + str(df['image'][i]) + "\t" + str(df['level'][i]) + "\t" + str(predictedVal[0]) + "\n"
	f.write(resultLine)

f.close()
