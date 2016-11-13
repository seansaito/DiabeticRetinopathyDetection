
# coding: utf-8

# In[24]:

#get_ipython().magic(u'matplotlib inline')
import matplotlib.pyplot as plt
import caffe
import os, csv
import numpy as np
from PIL import Image

new_width = new_height = 512

# Point this to the location of the training images
train_dir = "../retinopathy/train/"

fnames = os.listdir(train_dir)
fnames = np.random.choice(fnames, 3500)

im = Image.open("10_left.jpeg")
res = im.resize((new_width, new_height), Image.ANTIALIAS)
img = np.array(res)
#plt.imshow(img)
img = np.swapaxes(img, 0, 2)
print img.shape


# ## Caffe network construction

# In[25]:

# Change this to the name of the deploy prototxt file. For example, if you are using alexnet, then put
# deploy_alexnet.prototxt
model = "deploy_alexnet.prototxt"

# This shoud point to the corresponding caffemodel in results/
weights = "../alexnet/alexnet_iter_2000.caffemodel"

# Sets up the network
net = caffe.Net(str(model), str(weights), caffe.TEST)


# In[26]:

# We will take 10 sample images as that is the what the deploy prototxt requires
sample = np.random.choice(fnames, 10)
print "We have %i images" % len(fnames)
print sample


# In[27]:

#get_ipython().run_cell_magic(u'time', u'', u'sample_images = []\nfor f in sample:\n    im = Image.open(train_dir + f)\n    res = im.resize((new_width, new_height), Image.ANTIALIAS)\n    img = np.array(res)\n    img = np.swapaxes(img, 0, 2)\n    sample_images.append(img)')

sample_images = []

for f in sample:
    im = Image.open(train_dir + f)
    res = im.resize((new_width, new_height), Image.ANTIALIAS)
    img = np.array(res)
    img = np.swapaxes(img, 0, 2)
    sample_images.append(img)

# In[28]:

# Make sure shape is good
np.array(sample_images).shape


# In[29]:

im_input = np.array(sample_images)
net.blobs["data"].reshape(*im_input.shape)
net.blobs["data"].data[...] = im_input


# In[30]:

# get_ipython().run_cell_magic(u'time', u'', u'# Inference\nres = net.forward()')
res = net.forward()

# In[31]:

# This outputs the softmax probabilities
res["loss"]


# In[32]:

np.argmax(res["loss"], axis=1)


# In[ ]:

# get_ipython().run_cell_magic(u'time', u'', u'# Get the labels for all train images\nlabels = {}\nwith open("trainLabels.csv", "r") as fp:\n    reader = csv.DictReader(fp)\n    for row in reader:\n        labels[row["image"]] = int(row["level"])\n\n# Helper function\nfrom itertools import islice\ngroup_adjacent = lambda a, k: zip(*(islice(a, i, None, k) for i in range(k)))\ngroups = group_adjacent(fnames, 10)\n        \n# Write predictions to a CSV so that we can get the kappa score\n# The CSV has the following columns: image (image name), prediction, label (taken from trainLabels.csv)\nwith open("predictions.csv", "w") as fp:\n    fieldnames = ["image", "prediction", "label"]\n    writer = csv.DictWriter(fp, fieldnames=fieldnames)\n    \n    writer.writeheader()\n    for group in groups:\n        images = []\n        for f in group:\n            im = Image.open(train_dir + f)\n            res = im.resize((new_width, new_height), Image.ANTIALIAS)\n            img = np.array(res)\n            img = np.swapaxes(img, 0, 2)\n            images.append(img)\n        im_input = np.array(im_input)\n        net.blobs["data"].reshape(*im_input.shape)\n        net.blobs["data"].data[...] = im_input\n        res = net.forward()\n        prediction = np.argmax(res["loss"], axis=1)\n        for f, pred in zip(group, prediction):\n            fname = f.split(".")[0]\n            writer.writerow({"image": fname, "prediction": pred, "label": labels[fname]})')
# Get the labels for all train images
labels = {}
with open("trainLabels.csv", "r") as fp:
    reader = csv.DictReader(fp)
    for row in reader:
        labels[row["image"]] = int(row["level"])

# Helper function
from itertools import islice
group_adjacent = lambda a, k: zip(*(islice(a, i, None, k) for i in range(k)))
groups = group_adjacent(fnames, 10)

import datetime
# Logging
def log(text, fp=None):
    text = '[%s] [Retinopathy Inference] %s' % (str(datetime.datetime.now()), text)

    if fp is not None:
        fp.write(text)

    print text
        
# Write predictions to a CSV so that we can get the kappa score
# The CSV has the following columns: image (image name), prediction, label (taken from trainLabels.csv)
with open("predictions.csv", "w") as fp:
    fieldnames = ["image", "prediction", "label"]
    writer = csv.DictWriter(fp, fieldnames=fieldnames)
    
    writer.writeheader()
    for idx, group in enumerate(groups):
	log("Group %i out of %i" % (idx, len(groups)))
        images = []
        for f in group:
            im = Image.open(train_dir + f)
            res = im.resize((new_width, new_height), Image.ANTIALIAS)
            img = np.array(res)
            img = np.swapaxes(img, 0, 2)
            images.append(img)
        im_input = np.array(im_input)
        net.blobs["data"].reshape(*im_input.shape)
        net.blobs["data"].data[...] = im_input
        res = net.forward()
        prediction = np.argmax(res["loss"], axis=1)
        for f, pred in zip(group, prediction):
            fname = f.split(".")[0]
            writer.writerow({"image": fname, "prediction": pred, "label": labels[fname]})

# In[ ]:



