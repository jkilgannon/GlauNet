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
# Eclipse -> CSC620-Thesis-Python -> glaunet_predict.py

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

path_model = '/worksite/'
out_path = '/outgoing/'
input_size = (160, 160, 3)
preferred_size = (160, 160)

#####################################

# custom_loss and soft_loss: Used by the model when loading
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

def jaccard_distance(y_true, y_pred):
    # y_true: ground truth.  y_pred: predictions
    #
    # This version uses the Jaccard loss (IOU) described in https://arxiv.org/pdf/1801.05746.pdf
    # Returns the Jaccard Distance.

    # y_true is 1 for correct pixels, and 0 for incorrect ones

    # https://github.com/keras-team/keras-contrib/blob/master/keras_contrib/losses/jaccard.py
    # Jaccard = (|X & Y|)/ (|X|+ |Y| - |X & Y|)
    #         = sum(|A*B|)/(sum(|A|)+sum(|B|)-sum(|A*B|))
    y_true = tf.reshape(y_true, (-1, input_size[0], input_size[1]))
    intersection = tf.reduce_sum(K.abs(y_true * y_pred))
    sum_ = tf.reduce_sum(K.abs(y_true) + K.abs(y_pred))
    jac = intersection / (sum_ - intersection)
    return (1 - jac)


def main():
    precision_all = []
    recall_all = []
    F_measure_all = []
    jaccard_all = []

    # File where we output the data
    output_file = out_path + 'correctness_measures_all_annotators.txt'
    outf = open(output_file, "w")

    for annotator in range(1,7):
        path_fundus = '/local/repository/training_data/other_annotators/' + str(annotator) +  '/fundusvalidate/'
        path_annotated = '/local/repository/training_data/other_annotators/' + str(annotator) +  '/maskvalidate1/'
        
        files = os.listdir(path_fundus)
        files.sort()
    
        ################################################################
        # Predict how this annotator would annotate the image, given
        # the learned model for that annotator.
        K.clear_session()
    
        monitor_type = 'loss'
        loss = custom_loss
        
        in_model_loc = path_model + 'model_' + str(annotator) + '.h5'
    
        model = tf.keras.models.load_model(in_model_loc,
               custom_objects={'soft_loss': soft_loss, 'custom_loss': custom_loss})
    
        # Lists of the precisions, recalls, and F's for all images for this annotator
        precision_list = []
        recall_list = []
        F_measure_list = []   
        jaccard_list = []     
    
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
    
                ###########################################################
                # Save off a copy of the predicted annotation.
                predicted2 = np.reshape(predicted, preferred_size)
                img = np.zeros((preferred_size[0], preferred_size[1], 3), dtype=np.uint8)
                
                empty_color= [255,255,255]   # white; used for the "not the right class" color.
                class_color = [0, 0, 0]      # black; color to make the pixels marked as being of the given class
                
                for x in range(preferred_size[0]):
                    for y in range(preferred_size[1]):
                        if predicted2[x,y] == 1.0:
                            img[x,y] = class_color
                        else:
                            img[x,y] = empty_color
                
                img_file = Image.fromarray(img)            
                img_file.save(out_path + f + "_predicted_mask_annotator-" + str(annotator) + ".png")
                
                ###########################################################
                # Calculate how "off" the image is.
                
                # Open the ground truth file.
                ground_truth_file = path_annotated + f + '.npy'
                ground_truth = np.load(ground_truth_file)
                ground_truth = ground_truth.reshape(preferred_size)
    
                # Compare ground truth to prediction.  Make a mask showing
                # where the pixels are wrong.
                incorrect_pixels = 0
                
                true_negatives = 0
                true_positives = 0
                false_negatives = 0
                false_positives = 0
                
                img = np.zeros((preferred_size[0], preferred_size[1], 3), dtype=np.uint8)
                incorrect_color = [255, 0, 0]      # red; color to make the pixels marked as being wrong
                for x in range(preferred_size[0]):
                    for y in range(preferred_size[1]):
                        if predicted2[x,y] == ground_truth[x,y]:
                            img[x,y] = empty_color
                            # Differentiate between true negatives and true positives
                            if ground_truth[x,y] == 1:
                                true_positives = true_positives + 1
                            else:
                                true_negatives = true_negatives + 1
                        else:
                            incorrect_pixels = incorrect_pixels + 1
                            img[x,y] = incorrect_color
                            # Differentiate between false negatives and false positives
                            if ground_truth[x,y] == 1:
                                false_positives = false_positives + 1
                            else:
                                false_negatives = false_negatives + 1
                            
                            
                img_file = Image.fromarray(img)            
                img_file.save(out_path + f + "_incorrect_annotator-" + str(annotator) + ".png")
                
                
                # Print out the ground truth image.
                img = np.zeros((preferred_size[0], preferred_size[1], 3), dtype=np.uint8)
                correct_color = [0, 255, 0]      # green; color to make the pixels marked as being right
                for x in range(preferred_size[0]):
                    for y in range(preferred_size[1]):
                        if ground_truth[x,y] == 1.0:
                            img[x,y] = correct_color
                        else:
                            img[x,y] = empty_color
                            
                img_file = Image.fromarray(img)            
                img_file.save(out_path + f + "_ground-truth_annotator-" + str(annotator) + ".png")
    
                # R Burns, Data Science, 11-Evaluation.pptx
                precision_val = true_positives / float(true_positives + false_positives)    # p
                recall_val = true_positives / float(true_positives + false_negatives)       # r
                F_measure = (2 * true_positives) / float(2 * true_positives + false_positives + false_negatives)
                #jaccard = jaccard_distance(ground_truth, predicted2)
    
                outf.write(f + "," + str(annotator) + ", Incorrect pixels: " + str(incorrect_pixels) + '\n')
                outf.write(f + "," + str(annotator) + ", Correct pixels: " + str((preferred_size[0] * preferred_size[1]) - incorrect_pixels) + '\n')
                outf.write(f + "," + str(annotator) + ", True positives: " + str(true_positives) + '\n')    # True positives
                outf.write(f + "," + str(annotator) + ", False positives: " + str(false_positives) + '\n')  # False positives
                outf.write(f + "," + str(annotator) + ", True negatives: " + str(true_negatives) + '\n')    # True negatives
                outf.write(f + "," + str(annotator) + ", False negatives: " + str(false_negatives) + '\n')  # False negatives
                outf.write(f + "," + str(annotator) + ", Precision (p): " + str(precision_val) + '\n')      # Precision
                outf.write(f + "," + str(annotator) + ", Recall (r): " + str(recall_val) + '\n')            # Recall
                outf.write(f + "," + str(annotator) + ", F-measure (F-sub-1): " + str(F_measure) + '\n')    # F-measure
                #outf.write(f + "," + str(annotator) + ", Jaccard Distance: " + str(jaccard) + '\n')    # F-measure
                                
                precision_list.append(precision_val)
                recall_list.append(recall_val)
                F_measure_list.append(F_measure)
                #jaccard_list.append(jaccard)
                
                precision_all.append(precision_val)
                recall_all.append(recall_val)
                F_measure_all.append(F_measure)
                #jaccard_all.append(jaccard)
                
        # Calculate means, medians of p, r, and F_1 for this annotator
        mean_precision = np.mean(precision_list)
        mean_recall = np.mean(recall_list)
        mean_F_measure = np.mean(F_measure_list)
        #mean_jaccard = np.mean(jaccard_list)
        
        median_precision = np.median(precision_list)
        median_recall = np.median(recall_list)
        median_F_measure = np.median(F_measure_list)
        #median_jaccard = np.median(jaccard_list)
        
        outf.write(str(annotator) + ", MEAN Precision (p): " + str(mean_precision) + '\n')        # Precision
        outf.write(str(annotator) + ", MEAN Recall (r): " + str(mean_recall) + '\n')              # Recall
        outf.write(str(annotator) + ", MEAN F-measure (F-sub-1): " + str(mean_F_measure) + '\n')  # F-measure
        #outf.write(str(annotator) + ", MEAN Jaccard Distance: " + str(mean_jaccard) + '\n')  
        
        outf.write(str(annotator) + ", MEDIAN Precision (p): " + str(median_precision) + '\n')        # Precision
        outf.write(str(annotator) + ", MEDIAN Recall (r): " + str(median_recall) + '\n')              # Recall
        outf.write(str(annotator) + ", MEDIAN F-measure (F-sub-1): " + str(median_F_measure) + '\n')  # F-measure
        #outf.write(str(annotator) + ", MEDIAN Jaccard Distance: " + str(median_jaccard) + '\n')  
        
        print('----------------------------')
        print(str(annotator) + ", MEAN Precision (p): " + str(mean_precision))
        print(str(annotator) + ", MEAN Recall (r): " + str(mean_recall))
        print(str(annotator) + ", MEAN F-measure (F-sub-1): " + str(mean_F_measure))
        #print(str(annotator) + ", MEAN Jaccard Distance: " + str(mean_jaccard))
        
        print(str(annotator) + ", MEDIAN Precision (p): " + str(median_precision))
        print(str(annotator) + ", MEDIAN Recall (r): " + str(median_recall))
        print(str(annotator) + ", MEDIAN F-measure (F-sub-1): " + str(median_F_measure))
        #print(str(annotator) + ", MEDIAN Jaccard Distance: " + str(median_jaccard))
        
    
    all_mean_precision = np.mean(precision_all)
    all_mean_recall = np.mean(recall_all)
    all_mean_F_measure = np.mean(F_measure_all)
    #all_mean_jaccard = np.mean(jaccard_all)
    
    all_median_precision = np.median(precision_all)
    all_median_recall = np.median(recall_all)
    all_median_F_measure = np.median(F_measure_all)
    #all_median_jaccard = np.median(jaccard_all)
    
    outf.write("All annotators, MEAN Precision (p): " + str(all_mean_precision) + '\n')
    outf.write("All annotators, MEAN Recall (r): " + str(all_mean_recall) + '\n')
    outf.write("All annotators, MEAN F-measure (F-sub-1): " + str(all_mean_F_measure) + '\n')
    #outf.write("All annotators, MEAN Jaccard Distance: " + str(all_mean_jaccard) + '\n')
    
    outf.write("All annotators, MEDIAN Precision (p): " + str(all_median_precision) + '\n')
    outf.write("All annotators, MEDIAN Recall (r): " + str(all_median_recall) + '\n')
    outf.write("All annotators, MEDIAN F-measure (F-sub-1): " + str(all_median_F_measure) + '\n')
    #outf.write("All annotators, MEDIAN Jaccard Distance: " + str(all_median_jaccard) + '\n')

    outf.close()

    print('**************************************')
    print("All annotators, MEAN Precision (p): " + str(all_mean_precision))
    print("All annotators, MEAN Recall (r): " + str(all_mean_recall))
    print("All annotators, MEAN F-measure (F-sub-1): " + str(all_mean_F_measure))
    #print("All annotators, MEAN Jaccard Distance: " + str(all_mean_jaccard))
    
    print("All annotators, MEDIAN Precision (p): " + str(all_median_precision))
    print("All annotators, MEDIAN Recall (r): " + str(all_median_recall))
    print("All annotators, MEDIAN F-measure (F-sub-1): " + str(all_median_F_measure))
    #print("All annotators, MEDIAN Jaccard Distance: " + str(all_median_jaccard))
    
    print("Done")


# Launch!
main()
