# Cuts down the fundus images to just the part around the cup.
# Used to balance the classes.

from PIL import Image
import numpy as np
import sys

print("start")

img_path = 'E:\\!data\\college\\2020 Spring\\CSC620 - Deep Learning\\cropped\\fundus\\'
mask_path = 'E:\\!data\\college\\2020 Spring\\CSC620 - Deep Learning\\cropped\\annotations\\'
finished_mask_path = 'E:\\!data\\college\\2020 Spring\\CSC620 - Deep Learning\\cropped\\mask\\'
out_path = 'E:\\!data\\college\\2020 Spring\\CSC620 - Deep Learning\\cropped\\out\\'
out_path_img = out_path + 'fundus\\'
#out_path_mask = out_path + 'mask\\'

# One directory for each class's masks
maskloc_dir = []
maskloc_dir.append(out_path + "mask0\\")
maskloc_dir.append(out_path + "mask1\\")
maskloc_dir.append(out_path + "mask2\\")



#for imagenum in range(1,461):
#    for annotator in range(1,7):

for imagenum in range(174, 337):
    for annotator in range(3, 4):
        print(imagenum, ":", annotator)

        if imagenum == 457 and annotator == 3:
            continue 
        if imagenum == 455 and annotator == 2:
            continue 
        if imagenum == 450 and annotator == 2:
            continue 
        if imagenum == 449 and annotator == 6:
            continue 
        if imagenum == 443 and annotator == 4:
            continue 
        if imagenum == 440 and annotator == 4:
            continue 
        if imagenum == 438 and annotator == 4:
            continue 
        if imagenum == 427 and annotator == 6:
            continue 
        if imagenum == 427 and annotator == 2:
            continue 
        if imagenum == 385 and annotator == 2:
            continue 
        if imagenum == 382 and annotator == 4:
            continue 
        if imagenum == 382 and annotator == 2:
            continue 
        if imagenum == 380 and annotator == 6:
            continue 
        if imagenum == 379 and annotator == 6:
            continue 
        if imagenum == 378 and annotator == 4:
            continue 
        if imagenum == 375 and annotator == 6:
            continue 
        if imagenum == 373 and annotator == 2:
            continue 
        if imagenum == 371 and annotator == 1:
            continue 
        if imagenum == 368 and annotator == 4:
            continue 
        if imagenum == 359 and annotator == 4:
            continue 
        if imagenum == 353 and annotator == 4:
            continue 
        if imagenum == 351 and annotator == 4:
            continue 
        if imagenum == 347 and annotator == 4:
            continue 
        if imagenum == 336 and annotator == 4:
            continue 
        if imagenum == 330 and annotator == 4:
            continue 
        if imagenum == 303 and annotator == 4:
            continue 
        if imagenum == 301 and annotator == 2:
            continue 
        if imagenum == 277 and annotator == 6:
            continue 
        if imagenum == 275 and annotator == 6:
            continue 
        if imagenum == 272 and annotator == 6:
            continue 
        if imagenum == 235 and annotator == 1:
            continue 
        if imagenum == 215 and annotator == 4:
            continue 
        if imagenum == 208 and annotator == 5:
            continue 
        if imagenum == 195 and annotator == 1:
            continue  
        if imagenum == 188 and annotator == 4:
            continue  
        if imagenum == 172 and annotator == 6:
            continue        
        if imagenum == 163 and annotator == 5:
            continue
        
        if imagenum == 5 and annotator == 4:
            continue 
        if imagenum == 15 and annotator == 6:
            continue 
        if imagenum == 25 and annotator == 6:
            continue
        if imagenum == 26 and annotator == 2:
            continue
        if imagenum == 30 and annotator == 4:
            continue
        if imagenum == 31 and annotator == 6:
            continue
        if imagenum == 32 and annotator == 6:
            continue
        if imagenum == 55 and annotator == 3:
            continue
        if imagenum == 65 and annotator == 4:
            continue
        if imagenum == 71 and annotator == 6:
            continue
        if imagenum == 72 and annotator == 5:
            continue
        if imagenum == 73 and annotator == 6:
            continue
        if imagenum == 77 and annotator == 2:
            continue
        if imagenum == 77 and annotator == 5:
            continue
        if imagenum == 83 and annotator == 6:
            continue
        if imagenum == 124 and annotator == 4:
            continue
        if imagenum == 125 and annotator == 5:
            continue
        if imagenum == 127 and annotator == 1:
            continue
        if imagenum == 136 and annotator == 3:
            continue
        if imagenum == 141 and annotator == 6:
            continue
        if imagenum == 152 and annotator == 6:
            continue
        if imagenum == 161 and annotator == 4:
            continue
                                                                                                
        eye_file = 'image' + str(imagenum) + 'prime.tif'
        mask_file = 'image' + str(imagenum) + '-3.tif'  # Only annotator 3 this time, so has a uniform name
        finished_mask_file = 'image' + str(imagenum) + 'prime.bmp'

        eye_file_loc = img_path + eye_file
        mask_file_loc = mask_path + mask_file
        finished_mask_file_loc = finished_mask_path + finished_mask_file
        
        try:
            eye = Image.open(eye_file_loc)
        except:
            print("Eye not found")
            continue

        try:
            mask_image = Image.open(mask_file_loc)
        except Exception as e:
            print("Mask " + mask_file_loc +  " not found")
            #print("Error: " + str(sys.exc_info()[0]))
            print("Error: " + str(e))
            continue
            
        try:
            finished_mask_image = Image.open(finished_mask_file_loc)
        except:
            print("Finished mask not found")
            continue
                
                
        eyeAry = np.array(eye)
        mask_imageAry = np.array(mask_image)
        
        # Make an array containing lots of zeroes plus the pixels of the annotation.
        diff = abs(eyeAry - mask_imageAry)
        
        # Get an array of n x m instead of the n x m x 3 of a TIFF image.
        # This adds the three layers of color together into one layer.
        # Note that x and y are transposed so this is now an m x n array!
        flattened_diff = diff.sum(-1)
        
        # Get the location of the top and bottom extremes of the outer circle.
        rows_not_all_zeroes = np.where(flattened_diff.any(axis = 1))[0]
        top_point = rows_not_all_zeroes[0]
        bottom_point = rows_not_all_zeroes[-1]
        
        # Get the location of the left and right extremes of the circle.
        cols_not_all_zeroes = np.where(flattened_diff.any(axis = 0))[0]
        left_point = cols_not_all_zeroes[0]
        right_point = cols_not_all_zeroes[-1]
        
        # Output a clipped portion of the fundus image
        img_crop_size = 160
        cropped_fundus = eye.crop((left_point, top_point, left_point + img_crop_size, top_point + img_crop_size))
        #cropped_fundus = eye.crop((left_point, top_point, right_point, bottom_point))  
        cropped_fundus.save(out_path_img + 'image' + str(imagenum) + 'prime.tif', compression='raw')
        
        # Output the same clipped portion of the finished mask as a numpy array
        cropped_mask = finished_mask_image.crop((left_point, top_point, left_point + img_crop_size, top_point + img_crop_size))
        #cropped_mask.save(out_path_mask + 'image' + str(imagenum) + 'prime.bmp')
        
        # NOTE: Will need to rename the numpy array with the extension .tif for parallelism with the name of the fundus images (not any more!)
        mask_array = np.asarray(cropped_mask)
        
        # Save off a mask file for each class, in which the class's pixels are 
        #   marked as 1 and other classes are marked as 0.
        for cls in range(3):
            np_mask = np.zeros((img_crop_size, img_crop_size), dtype=np.uint8)
            
            # Not pretty, but easy to prove it properly marks the classes.
            for row in range(img_crop_size):
                for col in range(img_crop_size):
                    if cls == 0 and mask_array[row,col] == 0:
                        np_mask[row, col] = 1
                    elif cls == 1 and mask_array[row,col] == 1:
                        np_mask[row, col] = 1
                    elif cls == 2 and mask_array[row,col] == 2:
                        np_mask[row, col] = 1
                        
            np.save(maskloc_dir[cls] + 'image' + str(imagenum) + 'prime.tif', np_mask)
        
        

print("done")

