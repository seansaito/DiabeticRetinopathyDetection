from PIL import Image

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
