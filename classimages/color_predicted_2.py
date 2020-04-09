import numpy as np
from PIL import Image

input_shape=(160,160,3)

predicted = np.genfromtxt('predicted.csv', delimiter=',')
predicted = np.reshape(predicted, input_shape)

img = np.zeros((input_shape), dtype=np.uint8)

left_color = [0, 70, 20]
middle_color = [50, 0, 180]
right_color = [200, 100, 0]

for x in range(input_shape[0]):
    for y in range(input_shape[1]):
        if predicted[x,y,0] == 1.0:
            img[x,y] = left_color
        elif predicted[x,y,1] == 1.0:
            img[x,y] = middle_color
        else:
            img[x,y] = right_color

img_file = Image.fromarray(img)
img_file.save('predicted.png')

print("done")


