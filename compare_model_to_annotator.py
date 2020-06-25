from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from tensorflow.keras import backend as K
import os
import numpy as np
import tensorflow as tf
from PIL import Image
from random import randint
import math
from tensorflow.keras.losses import binary_crossentropy
import sys
from tensorflow.keras.utils import normalize
import copy


###############################################
########### CODE NOT USED.
########### DID THIS BY HAND INSTEAD.
###############################################

# Previous version(s): https://github.com/jkilgannon/GlauNet/blob/master/nn/onevsmany_LR_WORKING_glaunet.py 
#                      https://github.com/jkilgannon/GlauNet/blob/master/thesis/reload_model.py

"""
if len(sys.argv) < 3:
    # There must be a command line argument with the class and file name to use.
    # First CL arg (element zero) is the name of this program.
    print('This program requires two command line arguments: file name of the saved model file, and class (0..2), in that order. Exiting.')
    sys.exit()

print("==================================")
print("arg 1: " + sys.argv[1])
print("arg 2: " + sys.argv[2])
print("==================================")

# Which class are we training this time?
class_active = int(sys.argv[2])
if class_active < 0 or class_active > 2:
    print('This program requires two command line arguments: file name of the saved model file, and class (0..2), in that order. Exiting.')
    sys.exit()

# Location of the incoming saved model file
in_model_loc = sys.argv[1]
if not os.path.isfile(in_model_loc):
    print('The file ' + in_model_loc + ' must exist.  If no path was given, it must be in the directory with this executable. Exiting.')
    sys.exit()

print("==================================")
print("model file: " + in_model_loc)
print("==================================")
"""

path_fundus = './'
path_mask = './model_compare/mask/'
out_path = '/outgoing/'
input_size = (160, 160, 3)


"""
# Get the number of fundus image files and ground-truth mask files.
batchpath, batchdirs, fundus_files = next(os.walk(path_fundus))
num_fundus = len(fundus_files)
batchpath, batchdirs, mask_files = next(os.walk(path_mask))
num_mask = len(mask_files)

# now check that there are the same number of fundus and mask images
if num_fundus != num_mask:
   print("Number of fundus images does not match number of mask images. Exiting.")
   sys.exit()

print("Fundus files:")
print(fundus_files)
"""

#####################################

def custom_loss(y_true, y_pred):
    # y_true: ground truth.  y_pred: predictions
    #
    # This version uses the Jaccard loss (IOU) described in https://arxiv.org/pdf/1801.05746.pdf

    # y_true is 1 for correct pixels, and 0 for incorrect ones

    # https://github.com/keras-team/keras-contrib/blob/master/keras_contrib/losses/jaccard.py
    # Jaccard = (|X & Y|)/ (|X|+ |Y| - |X & Y|)
    #         = sum(|A*B|)/(sum(|A|)+sum(|B|)-sum(|A*B|))
    y_true = tf.reshape(y_true, (-1, input_size[0], input_size[1], 1))
    smooth = 100
    intersection = tf.reduce_sum(K.abs(y_true * y_pred))
    sum_ = tf.reduce_sum(K.abs(y_true) + K.abs(y_pred))
    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    return (1 - jac) * smooth

def soft_loss(y_true, y_pred):
    return 1 - custom_loss(y_true, y_pred)

"""
# A Python generator that will give the fit_generator data in batches.
def batchmaker(raw_loc, annotated_loc, batchsize, input_size):
    # Presumption: The two directories (raw and annotated) contain files with
    #   the same names, and the same number of files.
    #
    # raw_loc: path to the unannotated data
    # annotated_loc: path to the annotated data
    # batchsize: how much data do we want at a time?

    # Make a list of the file names. This will be the list for both dirs.
    files = os.listdir(raw_loc)
    file_size = len(files) - 1

    # The infinite loop is part of how generators work.  The fit_generator needs to
    # always have data available, so we loop forever.
    while True:
        # Randomize the list order.
        np.random.shuffle(files)

        # (Re)start at the head of the files.
        batch_head = 0           # Start of current batch in the files
        batch_end = batchsize    # End of the batch

        while batch_end < file_size:
            img = raw_loc + files[batch_head]
            target = annotated_loc + files[batch_head] + '.npy'

            test_fundus = load_img(img, target_size=input_size)
            x = image.img_to_array(test_fundus)

            # Get the target data, which is a saved numpy ndarray
            y = np.load(target)

            batch_head = batch_end + 1
            batch_end = batch_head + batchsize

            x_set = x.reshape((-1, input_size[0], input_size[1], input_size[2]))
            x_set = normalize(x_set)

            y_set = y.reshape((-1, input_size[0], input_size[1]))

            yield (x_set, y_set)
"""

def main():
    
    #for annotator in range(1,7):
    for annotator in range(1,2):
        K.clear_session()
    
        monitor_type = 'loss'
        loss = custom_loss
        
        in_model_loc = 'model_' + str(annotator) + '.h5'
    
        model = tf.keras.models.load_model(in_model_loc,
               custom_objects={'soft_loss': soft_loss, 'custom_loss': custom_loss})
               
        # Place the fundus TIFs in the directory with this program.
        img = 'test_' +str(annotator) + '.tif'
        test_fundus = load_img(img, target_size=input_size)
        
        # https://stackoverflow.com/questions/43469281/how-to-predict-input-image-using-trained-model-in-keras
        x = image.img_to_array(test_fundus)
        x = np.reshape(x, input_size)
        images = np.reshape(x, (-1, input_size[0], input_size[1], input_size[2]))
        # end of stackoverflow
        
        predicted = model.predict(images)
        
        ###
        
        # Get an array of n x m instead of the 1 x n x m x 1 of a predicted image
        # This adds the three layers of color together into one layer.
        # Note that x and y are transposed so this is now an m x n array! <<<<<<
        predicted = predicted.sum(-1)

        # Get the location of the top and bottom extremes of the outer circle.
        top_point = 0
        for row in predicted[0]:
            tmp = list(map(lambda x: x > 0.0, row))
            print(tmp)
            if any(tmp):
                # One of the elements in the row is not zero, so it's the top of the mask
                break
            top_point += 1

        #print("&&&&&&&&&&&&&&&&&&&&&&&&&&")

        bottom_point = len(predicted[0]) - 1
        while bottom_point >= 0:
            tmp = list(map(lambda x: x > 0.0, predicted[0][bottom_point]))
            #print(tmp)
            if any(tmp):
                # One of the elements in the row is not zero, so it's the bottom of the mask
                break        
            bottom_point =- 1 
            
        """            
        # Get the location of the top and bottom extremes of the outer circle.
        top_point = 0
        for row in predicted[0]:
            tmp = list(map(lambda x: x > 0.0, row))
            if any(tmp):
                # One of the elements in the row is not zero, so it's the top of the mask
                break
            top_point = top_point + 1

        #print("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")

        bottom_point = len(predicted[0]) - 1
        found_bot = False
        while bottom_point >= 0 and not found_bot:
            tmp = list(map(lambda x: x > 0.0, predicted[0][bottom_point]))
            found_bot = any(tmp)
            if not found_bot:
                bottom_point = bottom_point - 1 
        """

        # Get the location of the top and bottom extremes of the outer circle.
        #top_found = False
        #for x in range(len(predicted[0])):
        #    for y in range(len(predicted[0,0])):
                
        
        
        # Get the location of the top and bottom extremes of the outer circle.        
        top_point = 0
        bottom_point = 0
        row_num = 0
        top_found = False

        for row in predicted[0]:
            tmp = list(map(lambda x: x > 0.0, row))
            print(tmp)
            if any(tmp) and not top_found:
                # One of the elements in the row is not zero, so it's the top o$
                top_point = row_num
                top_found = True
            if not any(tmp) and top_found:
                bottom_point = row_num
                break
            row_num = row_num + 1

        # Get the location of the left and right extremes of the outer circle.
        for y in range(len(predicted[0])):
            for x in range(len(predicted[0])):
                # Both x and y are 160.
                if predicted[0,x,y] > 0.0:
                    # Found the left edge of the mask
                    left_point = y
                    
        for y in reversed(range(len(predicted[0]))):
            for x in range(len(predicted[0])):
                # Both x and y are 160.
                if predicted[0,x,y] > 0.0:
                    # Found the right edge of the mask
                    right_point = y
                    
        
        
        #map(lambda x: x > 0.0, row)
        #result = map(lambda x: x + x, numbers) 
        #top_point = 0
        #for row in predicted[0]:
        #    if row.any():
        #        print("found one!")
        #        break
        #    top_point += 1
        
        #bottom_point = len(predicted[0]) - 1
        #while bottom_point >= 0 and not predicted[0][bottom_point].any():
        #    bottom_point =- 1 
                
        
        #rows_not_all_zeroes = np.where(predicted.any(axis = 1))[0]
        #top_point = rows_not_all_zeroes[0]
        #bottom_point = rows_not_all_zeroes[-1]
        
        # Get the location of the left and right extremes of the circle.

        """        
        right_point = len(predicted[0]) - 1
        found_rt = False
        while right_point >= 0 and not found_rt:
            tmp = list(map(lambda x: x > 0.0, predicted[right_point][0]))
            found_rt = any(tmp)
            if not found_rt:
                right_point = right_point - 1         
        """
        
        #cols_not_all_zeroes = np.where(predicted.any(axis = 0))[0]
        #left_point = cols_not_all_zeroes[0]
        #right_point = cols_not_all_zeroes[-1]
    
        #print('==========================')
        #print(predicted[0])
        #print('==========================')
    
        print(top_point)
        print(bottom_point)
        print(left_point)
        print(right_point)
        print("------------------------")
        
        # Get the optic cup, which is an inner circle.

        center_point = (round((bottom_point - top_point)/2, 0), round((bottom_point - top_point)/2, 0))
        
        # First, "paint out" the background class.
        cup_predicted = copy.deepcopy(predicted[0])
        

        
        
        ############


        input_shape=(160,160)
        
        predicted = np.reshape(predicted, input_shape)
        img = np.zeros((input_shape[0], input_shape[1], 3), dtype=np.uint8)
        
        empty_color= [255,255,255]   # white; used for the "not the right class" color.
        class_color = [3, 70, 20]    # What color to make the pixels marked as being of$
        
        for x in range(input_shape[0]):
            for y in range(input_shape[1]):
                if predicted[x,y] == 1:
                    img[x,y] = class_color
                else:
                    img[x,y] = empty_color
        img_file = Image.fromarray(img)
        
        img_file.save("predicted_" + "temp" + ".png")



    print("Done")


# Launch!
main()
