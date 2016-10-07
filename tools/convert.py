"""Command-line tool to crop images to square and convert images
Adapted and heavily modified from https://github.com/sveitser/kaggle_diabetic
"""

from __future__ import division, print_function
import os
from multiprocessing.pool import Pool

import click
import numpy as np
from PIL import Image, ImageFilter

from converter import adaptive_convert, max_convert, otsu_convert, li_convert


def get_destination_fname(fname, extension, destination):
    basename = os.path.splitext(os.path.basename(fname))[0]
    return os.path.join(destination, '{}.{}'.format(basename, extension))

def process(args):
    fun, arg = args
    destination, fname, crop_size, stretch, extension = arg
    destination_fname = get_destination_fname(fname, extension, destination)
    if not os.path.exists(destination_fname):
        img = fun(fname, crop_size, stretch=stretch)
        img.save(destination_fname, quality=100)

def should_convert(filename):
    return filename.endswith('jpeg') or filename.endswith('tiff')

@click.command()
@click.option('--source', default='data/train', show_default=True,
              help='Directory with original images.')
@click.option('--destination', default='data/train_res', show_default=True,
              help='Where to save converted images.')
@click.option('--method', default='adaptive', show_default=True,
              help='Method used to convert. Options are adaptive and max.')
@click.option('--crop_size', default=256, show_default=True,
              help='Size of converted images.')
@click.option('--stretch', default=True, show_default=True,
              help='Whether to stretch the cropped image or pad with black')
@click.option('--extension', default='tiff', show_default=True,
              help='Filetype of converted images.')
@click.option('--processors', default=4, show_default=True,
              help='Number of processors.')
def main(source, destination, method, crop_size, stretch, extension, processors):
    try:
        os.makedirs(destination)
    except OSError:
        pass

    if stretch == "True":
        stretch = True
    elif stretch == "False":
        stretch = False
    else:
        raise Exception('Stretch must be True or False')

    if method == 'max':
        convert_method = max_convert
    elif method == 'otsu':
        convert_method = otsu_convert
    elif method == 'li':
        convert_method = li_convert
    elif method == 'adaptive':
        convert_method = adaptive_convert
    else:
        raise Exception('Undefined conversion method')

    filenames = [os.path.join(folder, fname) for folder, _, fnames in os.walk(source)
                 for fname in fnames if should_convert(fname)]
    filenames = sorted(filenames)

    print('Resizing images in {} to {}, this takes a while.'.format(source, destination))

    n = len(filenames)
    # process in batches, sometimes weird things happen with Pool on my machine
    batchsize = 100
    batches = (n + batchsize - 1) // batchsize
    pool = Pool(processors)

    args = []

    for f in filenames:
        args.append((convert_method, (destination, f, crop_size, stretch, extension)))

    for i in range(batches):
        print('batch {:>2} / {}'.format(i + 1, batches))
        pool.map(process, args[i * batchsize: (i + 1) * batchsize])

    pool.close()

    print('done')

if __name__ == '__main__':
    main()
