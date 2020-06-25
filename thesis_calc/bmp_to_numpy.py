import os
import numpy as np
#import PIL.Image as Image
import cv2

# Takes in BMP files of arbitrary size, converts them to numpy arrays.  Will 
# attempt to convert every file in the in_path directory.
#
# This ONLY cares about class 1.  It won't convert class 0 or 2.

print("Started")

in_path = 'E:\\!data\\college\\2020 Spring\\CSC620 - Deep Learning\\test-holdout\\corrected_masks\\'
out_path = 'E:\\!data\\college\\2020 Spring\\CSC620 - Deep Learning\\test-holdout\\mask_npy\\'

files = os.listdir(in_path)

for f in files:
    input_file = in_path + f
    
    # Strip off "BMP" and append "tif.npy"
    npy_name = f[:-3] + 'tif.npy'
    
    output_file = out_path + npy_name
    
    #img = Image.open(input_file)
    #img_array= np.array(img)
    
    img_array = cv2.imread(input_file)
    
    rowsize = len(img_array)
    colsize = len(img_array[0])
    new_ary = np.ndarray(shape=(rowsize,colsize), dtype=np.uint8)

    for row in range(rowsize):
        for col in range(colsize):
            curr_element = img_array[row, col]
            
            # [255,255,255] is white, which is the background. The mask
            # is red, which is [255,0,0].
            if curr_element[0] == 255 and curr_element[1] == 255 and curr_element[2] == 255:
                new_ary[row,col] = 0
            else:
                new_ary[row,col] = 1

    new_ary.tofile(output_file)

    print("File converted: " + input_file)
    
print("Done")
