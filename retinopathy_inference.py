
# coding: utf-8

# In[24]:

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
# plt.imshow(img)
img = np.swapaxes(img, 0, 2)
print img.shape


# ## Caffe network construction

# In[25]:

# Change this to the name of the deploy prototxt file. For example, if you are using alexnet, then put
# deploy_alexnet.prototxt
model = "../lenet_deploy.prototxt"

# This shoud point to the corresponding caffemodel in results/
weights = "../lenet/lenet_iter_2000.caffemodel"

# Sets up the network
net = caffe.Net(str(model), str(weights), caffe.TEST)


# In[26]:

# We will take 10 sample images as that is the what the deploy prototxt requires
sample = np.random.choice(fnames, 10)
print "We have %i images" % len(fnames)
print sample


# In[27]:

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

# Inference
res = net.forward()


# In[31]:

# This outputs the softmax probabilities
res["loss"]


# In[32]:

np.argmax(res["loss"], axis=1)


# In[ ]:

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
        
# Write predictions to a CSV so that we can get the kappa score
# The CSV has the following columns: image (image name), prediction, label (taken from trainLabels.csv)
with open("lenet_predictions.csv", "w") as fp:
    fieldnames = ["image", "prediction", "label"]
    writer = csv.DictWriter(fp, fieldnames=fieldnames)
    
    writer.writeheader()
    for group in groups:
        images = []
        for f in group:
            im = Image.open(train_dir + f)
            res = im.resize((new_width, new_height), Image.ANTIALIAS)
            img = np.array(res)
            img = np.swapaxes(img, 0, 2)
            images.append(img)
        im_input = np.array(images)
        net.blobs["data"].reshape(*im_input.shape)
        net.blobs["data"].data[...] = im_input
        res = net.forward()
        prediction = np.argmax(res["loss"], axis=1)
        for f, pred in zip(group, prediction):
            fname = f.split(".")[0]
            writer.writerow({"image": fname, "prediction": pred, "label": labels[fname]})


# In[2]:

# get_ipython().system(u'jupyter nbconvert --to script "Retinopathy Inference.ipynb"')


# In[ ]:



