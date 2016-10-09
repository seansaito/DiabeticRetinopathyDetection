"""Command-line tool to rotate images
Adapted and heavily modified from https://github.com/sveitser/kaggle_diabetic
Sample usage:
python rotate.py --source=train/images --destination=train/rotated --rotations=5 --expand=False --crop_size=512

Comments:
expand=False on stretched images will crop some parts
expand=True will need to be re-cropped
"""

from __future__ import division, print_function
import os
from multiprocessing.pool import Pool

import click
import numpy as np
from PIL import Image, ImageFilter


def get_destination_fname(fname, theta, extension, destination):
    basename = os.path.splitext(os.path.basename(fname))[0]
    return os.path.join(destination, '{}_{}.{}'.format(basename, int(theta), extension))


def rotate(fname, theta, crop_size, expand=False):
    img = Image.open(fname)
    rotated = img.rotate(theta, expand=expand)
    return rotated.resize([crop_size, crop_size])


def process(args):
    destination, fname, rotations, expand, crop_size, extension = args
    for theta in xrange(0, 360, 360 // rotations):
        destination_fname = get_destination_fname(fname, theta, extension, destination)
        if not os.path.exists(destination_fname):
            img = rotate(fname, theta, crop_size, expand=expand)
            img.save(destination_fname, quality=100)


def should_convert(filename):
    return filename.endswith('jpeg') or filename.endswith('tiff')

@click.command()
@click.option('--source', default='data/train', show_default=True,
              help='Directory with original images.')
@click.option('--destination', default='data/train_res', show_default=True,
              help='Where to save converted images.')
@click.option('--rotations', default=6, show_default=True,
              help='How many rotations to do')
@click.option('--expand', default=False, show_default=True,
              help='Whether the image should expand (then re-scale)')
@click.option('--crop_size', default=256, show_default=True,
              help='Size of converted images.')
@click.option('--extension', default='tiff', show_default=True,
              help='Filetype of converted images.')
@click.option('--processors', default=4, show_default=True,
              help='Number of processors.')
def main(source, destination, rotations, expand, crop_size, extension, processors):
    try:
        os.makedirs(destination)
    except OSError:
        pass

    if expand == "True":
        expand = True
    elif expand == "False":
        expand = False
    else:
        raise Exception('Stretch must be True or False')


    filenames = [os.path.join(folder, fname) for folder, _, fnames in os.walk(source)
                 for fname in fnames if should_convert(fname)]
    filenames = sorted(filenames)

    print('Rotating images in {} to {} {} times, this takes a while.'.format(source, rotations, destination))

    n = len(filenames)
    # process in batches, sometimes weird things happen with Pool on my machine
    batchsize = 10
    batches = (n + batchsize - 1) // batchsize
    pool = Pool(processors)

    args = []

    for f in filenames:
        args.append((destination, f, rotations, expand, crop_size, extension))

    for i in range(batches):
        print('batch {:>2} / {}'.format(i + 1, batches))
        pool.map(process, args[i * batchsize: (i + 1) * batchsize])

    pool.close()

    print('done')

if __name__ == '__main__':
    main()
