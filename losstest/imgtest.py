import numpy as np


import PIL.Image as Image


img_path = '/worksite/inprime/all_data/'


out_path = ''



imagenum=333


input_file = img_path + 'image' + str(imagenum) + 'prime.tif'


output_file = out_path + 'image' + str(imagenum) + 'prime.tif'



img = Image.open(input_file)
img_array= np.array(img)
rowsize = len(img_array)
colsize = len(img_array[0])
# We'll find out all the possible values of the image's pixels.
valid_list = [hash(str(img_array[0,0]))]
element_list = [img_array[0,0]]
for row in range(rowsize):
        for col in range(colsize):
                if img_array[row,col][1] > 100:
                    print(str(img_array[row,col]))
                    input("ENTER to continue")

print("Done")
