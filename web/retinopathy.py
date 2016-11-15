import numpy as np
from PIL import Image, ImageFilter


def compute_score(cropped_img):
    return 0


def bbox_from_threshold_array(img, array):
    bbox = Image.fromarray(array).getbbox()
    if bbox == (0, 0, img.width, img.height):
        inverted = (array - 1) * -1
        return Image.fromarray(inverted).getbbox()
    return bbox


def crop(img, bbox, crop_size, stretch=True):
    cropped = img.crop(bbox)

    if stretch:
        return cropped.resize([crop_size, crop_size])
    else:
        size = max(cropped.size)
        fill_color = (0, 0, 0)
        image = Image.new('RGB', (size, size), fill_color)
        offset_x = (size - cropped.size[0]) / 2
        offset_y = (size - cropped.size[1]) / 2
        image.paste(cropped, (offset_x, offset_y))
        return image.resize([crop_size, crop_size])


def max_convert(img, crop_size, stretch=True):
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

        if bbox is not None:
            left, upper, right, lower = bbox
            # if we selected less than 80% of the original
            # height, just crop the square
            if right - left < 0.8 * h or lower - upper < 0.8 * h:
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
