"""This is an adaptive threshold-based conversion"""

import cv2
import numpy as np
from PIL import Image
from crop import bbox_from_threshold_array, crop

def adaptive_convert(fname, crop_size, stretch=True):
    img = Image.open(fname)
    rgb = np.array(img).mean(axis=2).astype(np.uint8) # Use the mean as a proxy for brightness

    # Adaptive threshold to find border
    thresholded = cv2.adaptiveThreshold(rgb, 1, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        thresholdType=cv2.THRESH_BINARY_INV, blockSize=11, C=0)
    # Median blur to erode garbage
    thresholded = cv2.medianBlur(thresholded, 9)

    bbox = bbox_from_threshold_array(img, thresholded)
    if bbox == (0, 0, img.width, img.height):
        print "Could not find bbox for {}".format(fname)

    return crop(img, bbox, crop_size, stretch=stretch)
