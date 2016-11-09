import caffe
import lmdb
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os, csv

# Change these accordingly
train_dir = "/home/users/nus/a0109194/payton/e0046656/retinopathy/train/max-cropped-rotated/" 
test_dir = "/home/users/nus/a0109194/payton/e0046656/retinopathy/test/max-cropped-rotated/"
train_labels_f = "/home/users/nus/a0109194/payton/e0046656/retinopathy/train/labels.csv"
test_labels_f = "/home/users/nus/a0109194/payton/e0046656/retinopathy/test/labels.csv"

# Load labels
fp = open(train_labels_f)
reader = csv.DictReader(fp)
train_labels = {row["image"]: int(row["level"]) for row in reader}
fp.close()

fp = open(test_labels_f)
reader = csv.DictReader(fp)
test_labels = {row["image"]: int(row["level"]) for row in reader}
fp.close()
print "Done reading labels"

# File names
train_image_f = os.listdir(train_dir)
test_image_f = os.listdir(test_dir)

# Now create the lmdbs
def write_to_lmdb(lmdb_name, fnames, dir, labels=None, write_label=True):
	print "Writing lmdb %s" % lmdb_name
	map_size = len(fnames) * (2**20) * 1e1
	env = lmdb.open(lmdb_name, map_size=map_size)

	with env.begin(write=True) as txn:
		for idx, fname in enumerate(fnames):
			without_extension = fname.split(".")[0]
			if len(without_extension.split("_")) > 2:
				without_extension = "_".join(without_extension.split("_")[:2])
			im = Image.open(dir + fname)
			img = np.array(im)
			if labels is not None:
				label = labels[without_extension]
			datum = caffe.proto.caffe_pb2.Datum()
			datum.channels = img.shape[2]
			datum.width = img.shape[1]
			datum.height = img.shape[0]
			datum.data = np.array(img).tobytes()
			if write_label:
				datum.label = int(label)
			str_id = "{:08}".format(idx)
			txn.put(str_id, datum.SerializeToString())
	print "Done writing %i entries to lmdb" % len(fnames)

write_to_lmdb("max_cropped_rotated_train_lmdb", train_image_f, train_dir, train_labels)
write_to_lmdb("max_cropped_rotated_test_lmdb", test_image_f, test_dir, test_labels)
