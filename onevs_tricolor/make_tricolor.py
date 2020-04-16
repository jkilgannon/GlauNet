# Creates a tricolor flag and a mask that matches it, to be used to test a neural network

from PIL import Image
import numpy
import random

"""
# It's easiest to just build an image on top of an existing one.
eye = Image.open("/worksite/fundus/image333prime.tif")
#eye = Image.open("E:\\!data\\college\\2020 Spring\\CSC620 - Deep Learning\\R\\messidor\\image333prime.jpg")
np_img = numpy.array(eye)
"""

# How big is the image?
width = 160
height = 160
#width = 1440
#height = 960

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
left_color = [0, 70, 20]
middle_color = [50, 0, 180]
right_color = [200, 100, 0]

for x in range(130):
    print(x)
    
    filename = "image" + str(x) + "prime.tif"
    #imgloc = "E:\\!data\\college\\2020 Spring\\CSC620 - Deep Learning\\tricolor\\img\\" + filename
    #maskloc = "E:\\!data\\college\\2020 Spring\\CSC620 - Deep Learning\\tricolor\\mask\\" + filename
    imgloc = "tricolor/img/" + filename
    maskloc = "tricolor/mask/" + filename
    
    # Vary the width of the bands a little
    variance = 15
    #variance = 70
    left_div_local = left_divider + random.randint(-variance, variance)
    right_div_local = right_divider + random.randint(-variance, variance)
    
    # Make an image of a tricolor flag with the required widths and colors.
    # At the same time, make a mask of the same tricolor flag.  Classes are 1, 1, 2.
    for row in range(height):
        for col in range(left_div_local):
            np_img[row, col] = left_color
            np_mask[row, col] = 0
        for col in range(left_div_local, right_div_local):
            np_img[row, col] = middle_color
            np_mask[row, col] = 1
        for col in range(right_div_local, width):
            np_img[row, col] = right_color
            np_mask[row, col] = 2

    tricolor_img = Image.fromarray(np_img)
    tricolor_img.save(imgloc)
    
    numpy.save(maskloc, np_mask)    

print("Done!")

