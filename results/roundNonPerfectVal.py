'''
Round off the SVM results to the nearest value.
'''
import sys

originalResult = sys.argv[1]

f = open(originalResult, 'r')

for line in f:
	lineArr = line.strip().split('\t')
	actualVal = lineArr[2]
	predictedVal = lineArr[3]
	roundedVal = int(round(min(max(float(predictedVal), 0),4)))
	lineArr[3] = roundedVal
	print "\t".join(map(str,lineArr))
