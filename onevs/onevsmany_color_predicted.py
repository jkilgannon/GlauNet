import numpy as np
from PIL import Image

input_shape=(160,160)

predicted = np.genfromtxt('predicted.csv', delimiter=',')
predicted = np.reshape(predicted, input_shape)

img = np.zeros((input_shape[0], input_shape[1], 3), dtype=np.uint8)

#left_color = [0, 70, 20]
#middle_color = [50, 0, 180]
#right_color = [200, 100, 0]

left_color = [3, 70, 20]
middle_color = [50, 14, 180]
right_color = [200, 100, 5]

empty_color= [255,255,255]   # white; used for the "not the right class" color.
class_color = left_color     # Which color is the one we were searching for?

for x in range(input_shape[0]):
    for y in range(input_shape[1]):
        if predicted[x,y] == 1:
            img[x,y] = class_color
        else:
            img[x,y] = empty_color

img_file = Image.fromarray(img)
img_file.save('predicted.png')

print("done")

