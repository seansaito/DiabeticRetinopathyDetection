"""This uses Otsu's thresholding to decide the bounding box
Otsu's thresholding looks for a fixed threshold that
minimizes the variance between the two classes
"""

import cv2
import numpy as np
from PIL import Image
from crop import bbox_from_threshold_array, crop


def otsu_convert(fname, crop_size, stretch=True):
    img = Image.open(fname)
    rgb = np.array(img).mean(axis=2).astype(np.uint8) # Use the mean as a proxy for brightness

    # Use Otsu thresholding
    blurred = cv2.GaussianBlur(rgb,(5,5),0)
    ret3, thresholded = cv2.threshold(blurred,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    bbox = bbox_from_threshold_array(img, thresholded)
    if bbox == (0, 0, img.width, img.height):
        print "Could not find bbox for {}".format(fname)

    return crop(img, bbox, crop_size, stretch=stretch)
