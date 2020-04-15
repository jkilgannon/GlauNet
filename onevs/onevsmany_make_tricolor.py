# Creates a tricolor flag and a mask that matches it, to be used to test a neural network.
# This version is used for the one-vs-many design.

from PIL import Image
import numpy
import random
import os

# How big is the image?
width = 160
height = 160
#width = 1440
#height = 960

# How many images/masks should go into the validation directories? The
#   answer is: 1/val_ratio.  So if val_ratio is 5, 1/5 of images/masks
#   will be used in validation.
val_ratio = 5

np_img = numpy.zeros((height, width, 3), dtype=numpy.uint8)
np_mask = numpy.zeros((height, width), dtype=numpy.uint8)
print(np_img.shape)
print(np_mask.shape)

# How wide should one of the three bands be?
band_width = width // 3

# Divider between left-and-middle bands, and middle-and-right bands.
#  We will move these around for each image, to get varying "flags"
left_divider = band_width
right_divider = band_width * 2

# The three colors of the bands
left_color = [3, 70, 20]
middle_color = [50, 14, 180]
right_color = [200, 100, 5]

# Where to output the files
imgloc_dir = "/worksite/fundus/"
imgloc_validate_dir = "/worksite/fundusvalidate/"

# One directory for each class's masks
maskloc_dir = []
maskloc_validate_dir = []
maskloc_dir.append("/worksite/mask0/")
maskloc_dir.append("/worksite/mask1/")
maskloc_dir.append("/worksite/mask2/")
maskloc_validate_dir.append("/worksite/maskvalidate0/")
maskloc_validate_dir.append("/worksite/maskvalidate1/")
maskloc_validate_dir.append("/worksite/maskvalidate2/")

# Make directories, if needed
try:
    for direc in maskloc_dir:
        os.mkdir(direc)
    for direc in maskloc_validate_dir:
        os.mkdir(direc)
    os.mkdir(imgloc_dir)
    os.mkdir(imgloc_validate_dir)
except FileExistsError:
    print("Some directories already exist.")

# Clear directories
for filename in os.listdir(imgloc_dir):
    file_path = os.path.join(imgloc_dir, filename)
    if os.path.isfile(file_path):
        os.remove(file_path)

for filename in os.listdir(imgloc_validate_dir):
    file_path = os.path.join(imgloc_validate_dir, filename)
    if os.path.isfile(file_path):
        os.remove(file_path)

for cls in range(3):
    for filename in os.listdir(maskloc_dir[cls]):
        file_path = os.path.join(maskloc_dir[cls], filename)
        if os.path.isfile(file_path):
            os.remove(file_path)

    for filename in os.listdir(maskloc_validate_dir[cls]):
        file_path = os.path.join(maskloc_validate_dir[cls], filename)
        if os.path.isfile(file_path):
            os.remove(file_path)

train_count = 0
val_count = 0

for imgnum in range(130):
        print(imgnum)
    
        filename = "image" + str(imgnum) + "prime.tif"

        # Vary the width of the bands a little
        variance = 15
        #variance = 70
        left_div_local = left_divider + random.randint(-variance, variance)
        right_div_local = right_divider + random.randint(-variance, variance)
    
        # Make an image of a tricolor flag with the required widths and colors.
        #   At the same time, make a mask of the same tricolor flag.  Classes are 1, 1, 2.
        #   We are running one-vs-many, so we will just mark the "wanted" class with 1
        #   and the "unwanted" classes with 0.
        for row in range(height):
            for col in range(left_div_local):
                np_img[row, col] = left_color
            for col in range(left_div_local, right_div_local):
                np_img[row, col] = middle_color
            for col in range(right_div_local, width):
                np_img[row, col] = right_color

        tricolor_img = Image.fromarray(np_img)
        if imgnum % val_ratio == 0:
            # Save every n-th image and mask into the validation directory.
            tricolor_img.save(imgloc_validate_dir + filename)
            val_count = val_count + 1
        else:
            # Save 4/5 images and masks into the main directories to be used for training.
            tricolor_img.save(imgloc_dir + filename)
            train_count = train_count + 1

        # Save off a mask file for each class, in which the class's pixels are 
        #   marked as 1 and other classes are marked as 0.
        for cls in range(3):
            np_mask = numpy.zeros((height, width), dtype=numpy.uint8)

            # Not pretty, but easy to prove it properly marks the classes.
            for row in range(height):
                for col in range(left_div_local):
                    if cls == 0:
                        np_mask[row, col] = 1
                for col in range(left_div_local, right_div_local):
                    if cls == 1:
                        np_mask[row, col] = 1
                for col in range(right_div_local, width):
                    if cls == 2:
                        np_mask[row, col] = 1

            if imgnum % val_ratio == 0:
                # Save every n-th image and mask into the validation directory.
                numpy.save(maskloc_validate_dir[cls] + filename, np_mask)
            else:
                # Save 4/5 images and masks into the main directories to be used for training.
                numpy.save(maskloc_dir[cls] + filename, np_mask)

"""
# Rename the save numpy masks so they have the same name as the fundus images
#   they refer to; this makes the NN training much easier.
os.system('rename "s/tif.npy/tif/" /worksite/mask0/*')
os.system('rename "s/tif.npy/tif/" /worksite/mask1/*')
os.system('rename "s/tif.npy/tif/" /worksite/mask2/*')
os.system('rename "s/tif.npy/tif/" /worksite/maskvalidate0/*')
os.system('rename "s/tif.npy/tif/" /worksite/maskvalidate1/*')
os.system('rename "s/tif.npy/tif/" /worksite/maskvalidate2/*')
"""

print("Training images: " + str(train_count))
print("Validation images: " + str(val_count))

print("Done!")

