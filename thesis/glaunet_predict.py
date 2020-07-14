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
import sys
from tensorflow.keras.utils import normalize
#from random import randint
#import math
#from tensorflow.keras.losses import binary_crossentropy
#import copy

# Previous versions:
# Eclipse -> CSC620-Thesis-Python -> compare_model_to_annotators.py

# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#
# Remember! To get an element from an array, 
# use ary[y][x] since the "rows" are y and 
# the "columns" are x. Note that x and y
# are exactly as one expects in the Cartesian
# plane.
#
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


path_fundus = '/incoming/'
path_model = '/worksite/'
out_path = '/outgoing/'
input_size = (160, 160, 3)
preferred_size = (160, 160)

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


def main():
    # File where we output the data
    output_file = out_path + 'C_D_ratios.txt'
    outf = open(output_file, "w")
    
    # Place the fundus TIF in the <path_fundus> directory.
    files = os.listdir(path_fundus)
    files.sort()
    
    for annotator in range(1,7):
        ################################################################
        # Predict how each annotator would annotate the image, given
        # the learned model for that annotator.
        K.clear_session()
    
        monitor_type = 'loss'
        loss = custom_loss
        
        in_model_loc = path_model + 'model_' + str(annotator) + '.h5'
    
        model = tf.keras.models.load_model(in_model_loc,
               custom_objects={'soft_loss': soft_loss, 'custom_loss': custom_loss})

        for f in files:
            if f.lower().endswith('.tif'):
                img = path_fundus + f
                test_fundus = load_img(img, target_size=input_size)
                
                # https://stackoverflow.com/questions/43469281/how-to-predict-input-image-using-trained-model-in-keras
                x = image.img_to_array(test_fundus)
                x = np.reshape(x, input_size)
                images = np.reshape(x, (-1, input_size[0], input_size[1], input_size[2]))
                # end of stackoverflow
                           
                predicted = model.predict(images)
                predicted_reshape = np.reshape(predicted, preferred_size)
                
                ################################################################
                # Calculate the points used to get the C/D ratio.
                
                # Get the location of the top and bottom extremes of the disc.
                top_point = 0       # Darn local variables inside for loops...
                bottom_point = 0        
                for x in range(160):
                    row_sum = 0
                    for y in range(160):
                        row_sum = row_sum + predicted_reshape[x][y]
                    if row_sum > 2:
                        top_point = x
                        break
        
                found_pt = False
                for x in range(159, top_point, -1):
                    row_sum = 0
                    for y in range(160):
                        row_sum = row_sum + predicted_reshape[x][y]
                    if row_sum > 2:
                        bottom_point = x
                        break
        
                # Get the location of the left and right extremes of the disc.
                left_point = 0 
                right_point = 0
                for y in range(160):
                    col_sum = 0
                    for x in range(160):
                        col_sum = col_sum + predicted_reshape[x][y]
                    if col_sum > 2:
                        left_point = y
                        break
                
                for y in range(159, left_point, -1):
                    col_sum = 0
                    for x in range(160):
                        col_sum = col_sum + predicted_reshape[x][y]
                    if col_sum > 2:
                        right_point = y
                        break
                
                disc_top_point = top_point
                disc_bottom_point = bottom_point
                disc_left_point = left_point
                disc_right_point = right_point
                
                disc_length = disc_bottom_point - disc_top_point
                disc_width = disc_right_point - disc_left_point
                
                # Get the optic cup, which is the inner circle.
                center_point = (int(round((right_point - left_point)/2, 0)), int(round((bottom_point - top_point)/2, 0)))
                
                # Find the top of the optic cup on the disc's center line.
                local_cup_top = 0
                for y in range(center_point[1], disc_top_point, -1):
                    if predicted_reshape[y][center_point[0]] > 0.0:
                        # Found the local top of the cup.
                        local_cup_top = y
                        break
                
                # Find the top pixels of the predicted optic cup.
                local_cup_top = disc_top_point + 1
                half_range = int(round((disc_width / 4), 0))
                x = center_point[0] - half_range
                top_list = []
                
                while x <= center_point[0] + half_range:
                    column_cup_top = 159
                    y = center_point[1]
                    stop_while = False
                    
                    while not stop_while and y > disc_top_point:
                        if predicted_reshape[y][x] > 0.0:
                            stop_while = True
                        else:
                            top_list.append(y)
                        
                        y = y - 1
                    
                    x = x + 1
                
                cup_top_point = min(top_list)
                top_list = []                   # Clear
                
                # Find the bottom pixels of the predicted optic cup.
                local_cup_bottom = disc_bottom_point - 1
                x = center_point[0] - half_range
                bottom_list = []
                
                while x <= center_point[0] + half_range:
                    column_cup_bottom = 0
                    y = center_point[1]
                    stop_while = False
                    
                    while not stop_while and y < disc_bottom_point:
                        if predicted_reshape[y][x] > 0.0:
                            stop_while = True
                        else:
                            bottom_list.append(y)
                        
                        y = y + 1
                    
                    x = x + 1
                
                cup_bottom_point = max(bottom_list)
                bottom_list = []                        # Clear
                
                
                ###########################################################
                # Calculate C/D ratio and output to file.
                cup_length = cup_bottom_point - cup_top_point
                disc_length = disc_bottom_point - disc_top_point
                C_D_ratio = float(cup_length) / float(disc_length)
                outf.write("Processing image: '" + f + "', Annotator " + str(annotator) + ": cup = " + str(cup_length) + ", disc = " + str(disc_length) + ", C/D ratio = " + str(C_D_ratio) + '\n')
                

                """                                      
                ###########################################################
                # Save off a copy of the predicted annotation.
                predicted2 = np.reshape(predicted, preferred_size)
                img = np.zeros((preferred_size[0], preferred_size[1], 3), dtype=np.uint8)
                
                empty_color= [255,255,255]   # white; used for the "not the right class" color.
                class_color = [3, 70, 20]    # What color to make the pixels marked as being of the given class
                
                for x in range(preferred_size[0]):
                    for y in range(preferred_size[1]):
                        if predicted2[x,y] == 1.0:
                            img[x,y] = class_color
                        else:
                            img[x,y] = empty_color
                img_file = Image.fromarray(img)
                
                img_file.save(f + "_predicted_mask_annotator-" + str(annotator) + ".png")
                #print("PNGs saved off")
                """

    outf.close()
    print("Done")


# Launch!
main()
