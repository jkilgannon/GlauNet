# Figures out how the pixels are stored in the segmented TIF files




import numpy as np


import PIL.Image as Image


img_path = '/users/jk880380/'


out_path = ''



input_file = img_path + 'predicted.csv'
# Open and save in one fell swoop
f = open(input_file, "r")
contents = f.read()
elements = contents.split(",")	
valid_list = []
for num in elements:
    h = hash(str(num))
    if not h in valid_list:
        valid_list.append(h)
        print(valid_list)

print("Done")
