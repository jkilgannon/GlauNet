import numpy as np
from PIL import Image

#input_shape=(160,160)
input_shape=(960, 1440)

predicted = np.genfromtxt('predicted.csv', delimiter=',')
predicted = np.reshape(predicted, input_shape)

img = np.zeros((input_shape[0], input_shape[1], 3), dtype=np.uint8)

left_color = [3, 70, 20]
middle_color = [50, 14, 180]
right_color = [200, 100, 5]

empty_color= [255,255,255]   # white; used for the "not the right class" color.
class_color = left_color     # Which color is the one we were searching for?

equals1 = 0

for x in range(input_shape[0]):
    for y in range(input_shape[1]):
        if predicted[x,y] == 1:
            img[x,y] = class_color
            equals1 = equals1 + 1
        else:
            img[x,y] = empty_color

img_file = Image.fromarray(img)
img_file.save('predicted.png')

print("Found " + str(equals1) + " instance(s) of the class with confidence 1.0")

##

img = np.zeros((input_shape[0], input_shape[1], 3), dtype=np.uint8)

gtzero = 0

for x in range(input_shape[0]):
    for y in range(input_shape[1]):
        if predicted[x,y] >= 0.00000001:
            img[x,y] = class_color
            gtzero = gtzero + 1
        else:
            img[x,y] = empty_color

img_file = Image.fromarray(img)
img_file.save('predicted_0.00000001.png')

print("Found " + str(gtzero) + " instance(s) of the class with confidence greater than 0.00000001")

##

print("done")
