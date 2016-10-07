"""This Li's Minimum Cross-Entropy to decide the threshold"""

import cv2
import numpy as np
from PIL import Image
import skimage.filters
from crop import bbox_from_threshold_array, crop


def li_convert(fname, crop_size, stretch=True):
    img = Image.open(fname)
    rgb = np.array(img).mean(axis=2).astype(np.uint8) # Use the mean as a proxy for brightness

    # Use Li's minimum cross entropy
    threshold = skimage.filters.threshold_li(rgb)
    _, thresholded = cv2.threshold(rgb, threshold, 255,cv2.THRESH_BINARY)

    bbox = bbox_from_threshold_array(img, thresholded)
    if bbox == (0, 0, img.width, img.height):
        print "Could not find bbox for {}".format(fname)

    return crop(img, bbox, crop_size, stretch=stretch)
