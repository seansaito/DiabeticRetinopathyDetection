from scoreResults import *
import pandas as pd
import sys


#quadratic_weighted_kappa(rater_a, rater_b)

#df = pd.read_csv(sys.argv[1])

scoringResults = sys.argv[1]

f = open(scoringResults, 'r')

actualValList = []
predictedValList = []

for line in f:
	lineArr = line.strip().split('\t')
	actualVal = lineArr[2]
	predictedVal = lineArr[3]
	actualValList.append(actualVal)
	predictedValList.append(predictedVal)

#print actualValList

print quadratic_weighted_kappa(actualValList, predictedValList)
