"""This is the convert method used in https://github.com/sveitser/kaggle_diabetic
They got second place at the Kaggle competition

Their method uses the fact that the left-most 1/32 and the
right-most 1/32 are background and black. They just find the
brightest spot there and use that as a fixed threshold.

If they run into problems, they just default to cropping
the left and right sides to make a square
"""

import numpy as np
from PIL import Image, ImageFilter
from crop import crop


def max_convert(fname, crop_size, stretch=True):
    img = Image.open(fname)

    # Blur the image
    blurred = img.filter(ImageFilter.BLUR)
    ba = np.array(blurred)
    h, w, _ = ba.shape

    if w > 1.2 * h:
        # Look at the left-most 1/32 and the right-most 1/32
        # and find the brightest spot
        left_max = ba[:, : w // 32, :].max(axis=(0, 1)).astype(int)
        right_max = ba[:, - w // 32:, :].max(axis=(0, 1)).astype(int)
        max_bg = np.maximum(left_max, right_max)

        # Everything a little brigher than the brightest spot in the background is foreground
        foreground = (ba > max_bg + 10).astype(np.uint8)
        bbox = Image.fromarray(foreground).getbbox()

        if bbox is None:
            # This is probably just debugging
            print('bbox none for {} (???)'.format(fname))
        else:
            left, upper, right, lower = bbox
            # if we selected less than 80% of the original 
            # height, just crop the square
            if right - left < 0.8 * h or lower - upper < 0.8 * h:
                print('bbox too small for {}'.format(fname))
                bbox = None
    else:
        bbox = None

    if bbox is None:
        bbox = square_bbox(img)

    return crop(img, bbox, crop_size, stretch=stretch)


def square_bbox(img):
    w, h = img.size
    left = max((w - h) // 2, 0)
    upper = 0
    right = min(w - (w - h) // 2, w)
    lower = h
    return (left, upper, right, lower)
