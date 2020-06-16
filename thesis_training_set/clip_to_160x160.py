# Takes the oversized masks that were made in 2019 and clips them to the correct size. Makes the
# cropped fundus and mask files, for all classes and annotators.

from PIL import Image
import numpy as np
import sys

print("start")


# img_path: to the original fundus image
# mask_path: to the original annotated image
# finished_mask_path: to the (oversized) segmented mask image
#
# img and mask are in the same folder.
img_path = 'E:\\!data\\college\\2020 Spring\\CSC620 - Deep Learning\\cropped\\raws\\'
mask_path = 'E:\\!data\\college\\2020 Spring\\CSC620 - Deep Learning\\cropped\\raws\\'
finished_mask_path = 'E:\\!data\\college\\2020 Spring\\CSC620 - Deep Learning\\glaucomaData\\segmented\\output\\'
out_path = 'E:\\!data\\college\\2020 Spring\\CSC620 - Deep Learning\\cropped\\out\\'


for imagenum in range(174, 337):
    for annotator in range(1, 7):
        if annotator == 3:
            # Annotator #3 is already done
            continue
        
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
        mask_file = 'image' + str(imagenum) + '-' + str(annotator) + '.tif'
        oversize_mask_file = 'segmented_image' + str(imagenum) + '-' + str(annotator) + '.bmp'
        
        eye_file_loc = img_path + eye_file
        mask_file_loc = mask_path + mask_file
        finished_mask_file_loc = finished_mask_path + oversize_mask_file
        
        try:
            eye = Image.open(eye_file_loc)
        except:
            print("Eye not found")
            continue

        try:
            mask_image = Image.open(mask_file_loc)
        except Exception as e:
            print("Mask " + mask_file_loc +  " not found")
            print("Error: " + str(e))
            continue
            
        try:
            finished_mask_image = Image.open(finished_mask_file_loc)
        except:
            print("Finished mask not found")
            continue
        
        # Trim the mask to the same size as the fundus image. The mask was made
        # to have the max size of the largest image, originally
        width, height = eye.size
        finished_mask_image = finished_mask_image.crop((0, 0, width, height)) 
        
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
        
        # 20% of the data will be used to validate, 80% to train.
        if imagenum % 5 == 0:
            new_fundus_save_loc = out_path + str(annotator) + '\\fundusvalidate\\'
        else:
            new_fundus_save_loc = out_path + str(annotator) + '\\fundus\\'
          
        cropped_fundus.save(new_fundus_save_loc + 'image' + str(imagenum) + 'prime.tif', compression='raw')
        
        # Output the same clipped portion of the finished mask as a numpy array
        cropped_mask = finished_mask_image.crop((left_point, top_point, left_point + img_crop_size, top_point + img_crop_size))
        
        mask_array = np.asarray(cropped_mask)

        # Test to make sure the mask image is saving sensibly.  We won't actually use this for anything
        # other than checking the data.
        cropped_mask.save(out_path + 'sanity_checks\\' + 'image' + str(imagenum) + '-annotator' + str(annotator) + 'TEST.tif', compression='raw')
        
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
                        
            # 20% of the data will be used to validate, 80% to train.
            if imagenum % 5 == 0:
                new_mask_save_loc = out_path + str(annotator) + '\\maskvalidate' + str(cls) + '\\'
            else:
                new_mask_save_loc = out_path + str(annotator) + '\\mask' + str(cls) + '\\'
            
            #print(new_mask_save_loc)
            
            np.save(new_mask_save_loc + 'image' + str(imagenum) + 'prime.tif', np_mask)

print("done")
