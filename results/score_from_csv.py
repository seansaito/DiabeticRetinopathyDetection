from scoreResults import *
import csv, os, sys

csv_file = sys.argv[1]

f = open(csv_file, "r")
labels, predictions = [], []

reader = csv.DictReader(f)
for row in reader:
    labels.append(row["label"])
    predictions.append(row["prediction"])

print quadratic_weighted_kappa(labels, predictions)
