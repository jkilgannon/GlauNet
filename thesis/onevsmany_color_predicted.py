import numpy as np
from PIL import Image
import sys

input_shape=(160,160)

predicted = np.genfromtxt('predicted.csv', delimiter=',')
predicted = np.reshape(predicted, input_shape)
img = np.zeros((input_shape[0], input_shape[1], 3), dtype=np.uint8)

empty_color= [255,255,255]   # white; used for the "not the right class" color.
class_color = [3, 70, 20]    # What color to make the pixels marked as being of the class

for x in range(input_shape[0]):
    for y in range(input_shape[1]):
        if predicted[x,y] == 1:
            img[x,y] = class_color
        else:
            img[x,y] = empty_color
img_file = Image.fromarray(img)

# Save off the image, with the epoch number embedded
epochnum = sys.argv[1]
print("epochnum: " + epochnum)
img_file.save("predicted_" + epochnum + ".png")

print("done")
