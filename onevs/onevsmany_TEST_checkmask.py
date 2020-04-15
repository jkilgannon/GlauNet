import numpy as np
from PIL import Image

file_wanted = 'image33prime.tif.npy'
input_size = (160, 160)

# The three colors of the bands
left_color = [3, 70, 20]
middle_color = [50, 14, 180]
right_color = [200, 100, 5]

color_wanted = [0,0,0]        # The color we want, masked as black.
color_wrong = [255,255,255]   # If this isn't the color we want, mask it out as white.

y = np.load(file_wanted)
y = y.reshape((-1, input_size[0], input_size[1]))
y.tofile('mask_readable.csv', sep=',')

np_img = np.zeros((input_size[0], input_size[1], 3), dtype=np.uint8)

for row in range(input_size[0]):
    for col in range(input_size[1]):
        if y[0, row, col] == 1:
            np_img[row, col] = color_wanted
        else:
            np_img[row, col] = color_wrong

mask_img = Image.fromarray(np_img)
mask_img.save('mask_readable.png')
