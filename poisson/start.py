import os
import errno
from os import path
from glob import glob
import cv2
import numpy as np
import poisson
import time
t0 = time.time()
IMG_EXTENSIONS = ["png", "jpeg", "jpg", "JPG", "gif", "tiff", "tif", "raw", "bmp"]
SRC_FOLDER = r"C:\Users\DELL\Desktop\CloudRemoval\poisson-image-editing-master\input\2"
OUT_FOLDER = r"output"


def collect_files(prefix, extension_list=IMG_EXTENSIONS):
    filenames = sum(map(glob, [prefix + ext for ext in extension_list]), [])
    return filenames


subfolders = os.walk(SRC_FOLDER)
print(subfolders)
# subfolders.next()

for dirpath, dirnames, fnames in subfolders:
    print(dirpath, dirnames, fnames)
    image_dir = os.path.split(dirpath)[-1]
    output_dir = os.path.join(OUT_FOLDER, image_dir)
    print("Processing input {i}...".format(i=image_dir))

    # Search for images to process
    source_names = collect_files(os.path.join(dirpath, '*source.'))
    target_names = collect_files(os.path.join(dirpath, '*target.'))
    mask_names = collect_files(os.path.join(dirpath, '*mask.'))

    if not len(source_names) == len(target_names) == len(mask_names) == 1:
        print("There must be one source, one target, and one mask per input.")
        continue

    # Read images
    source_img = cv2.imread(source_names[0], cv2.IMREAD_COLOR)
    target_img = cv2.imread(target_names[0], cv2.IMREAD_COLOR)
    mask_img = cv2.imread(mask_names[0], cv2.IMREAD_GRAYSCALE)

    source_img = cv2.copyMakeBorder(source_img, 1, 1, 1, 1, borderType=cv2.BORDER_CONSTANT, value=0)
    target_img = cv2.copyMakeBorder(target_img, 1, 1, 1, 1, borderType=cv2.BORDER_CONSTANT, value=0)
    mask_img = cv2.copyMakeBorder(mask_img, 1, 1, 1, 1, borderType=cv2.BORDER_CONSTANT, value=0)

    # Normalize mask to range [0,1]
    mask = np.atleast_3d(mask_img).astype(np.float16) / 255.
    # Make mask binary
    mask[mask != 1] = 0
    # Trim to one channel
    mask = mask[:, :, 0]
    channels = source_img.shape[-1]
    # Call the poisson method on each individual channel
    result_stack = [poisson.process(source_img[:, :, i], target_img[:, :, i], mask) for i in range(channels)]
    # Merge the channels back into one image
    result = cv2.merge(result_stack)
    # Make result directory if needed
    try:
        os.makedirs(output_dir)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise
    h, w, c = result.shape
    result = result[1:h-1, 1:w-1, :]

    # Write result
    cv2.imwrite(path.join(output_dir, 'result.png'), result)
    print("Finished processing input {i}.".format(i=image_dir))
    t1 = time.time()
    print('time:{}'.format(t1-t0))
