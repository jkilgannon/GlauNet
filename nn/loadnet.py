from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
#from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from tensorflow.keras import backend as keras
import os
import numpy as np
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt


def dice_coeff_inverted(y_true, y_pred, smooth=1e-7):
    return (1 - dice_coeff(y_true, y_pred, smooth))


# https://stackoverflow.com/questions/53248099/keras-image-segmentation-using-grayscale-masks-and-imagedatagenerator-class 
# https://github.com/keras-team/keras/issues/3611                                                                                                                                                          def 
#dice_coeff(y_true, y_pred):
def dice_coeff(y_true, y_pred, smooth=1e-7): 
    intersection = keras.sum(y_true * y_pred, axis=[1,2,3])
    union = keras.sum(y_true, axis=[1,2,3]) + keras.sum(y_pred, axis=[1,2,3])
    return keras.mean( (2. * intersection + smooth) / (union + smooth), axis=0)    

def dice_loss_2(y_true, y_pred):
    numerator = 2 * tf.reduce_sum(y_true * y_pred, axis=(1,2,3))
    denominator = tf.reduce_sum(y_true + y_pred, axis=(1,2,3))
    return tf.reshape(1 - numerator / denominator, (-1, 1, 1))

def inverted_dice_2(y_true, y_pred):
    return 1 - dice_loss_2(y_true, y_pred)

def dice_loss_6(y_true, y_pred):
    numerator = 2 * tf.reduce_sum(y_true * y_pred, axis=(1,2))
    denominator = tf.reduce_sum(y_true + y_pred, axis=(1,2))
    return tf.reshape(1 - numerator / denominator, (-1, 1, 1))

def inverted_dice_6(y_true, y_pred):
    return 1 - dice_loss_6(y_true, y_pred)

def manhattan_coeff(y_true, y_pred):
    # Pseudo-Manhattan distance between the "locations" of the ground truth and
    #  predicted arrays.  Came up with this one myself.

    y_true_real = tf.dtypes.cast(y_true, tf.float32)
    y_pred_real = tf.dtypes.cast(y_pred, tf.float32)

    numerators = tf.reduce_sum(tf.math.abs(y_true_real - y_pred_real), axis=(1,2))
    numerator = 3 * tf.reduce_mean(numerators)
    denominator = input_size[0] * input_size[1] * input_size[2]

    return tf.reshape(1 - (numerator / denominator), (-1, 1, 1, 1))

    #numerator = smoothing_factor + (2 * tf.reduce_sum(y_true_real * y_pred_real, axis=(1,2,3)))
    #denominator = smoothing_factor + tf.reduce_sum(y_true_real + y_pred_real, axis=(1,2,3))
    #return tf.reshape(numerator / denominator, (-1, 1, 1, 1))

def soft_loss(y_true, y_pred):
    return 1 - manhattan_coeff(y_true, y_pred)



    	
# UNet:
# https://github.com/zhixuhao/unet
num_fl = 9
num_fl_val = 9
#num_fl = 130
#num_fl_val = 32
out_path = '/outgoing/'

#batchsize = 3
batchsize = 1
#input_size = (960, 1440, 3)
input_size = (320, 480, 3)

# Load the model from file
prevmodelfile = 'best_model_incremental.h5'
#prevmodelfile = 'last_weights.h5'
print(' Loading model: ' + prevmodelfile)
#model = load_model(prevmodelfile)
#model = load_model(prevmodelfile, custom_objects={'iou_coeff': iou_coeff})
#model = load_model(prevmodelfile, custom_objects={'dice_coeff': dice_coeff, 'dice_coeff_inverted': dice_coeff_inverted})
#model = load_model(prevmodelfile, custom_objects={'dice_loss_2': dice_loss_2, 'inverted_dice_2': inverted_dice_2})
#model = load_model(prevmodelfile, custom_objects={'dice_loss_6': dice_loss_6, 'inverted_dice_6': inverted_dice_6})
model = load_model(prevmodelfile, custom_objects={'soft_loss': soft_loss, 'manhattan_coeff': manhattan_coeff})
print("++++++++++++++")
print(model.count_params())
print("++++++++++++++")
print(model.summary())
print("++++++++++++++")
os.system('free -m')
print("++++++++++++++")
os.system('vmstat -s')
print("++++++++++++++")

img = 'test.tif'
test_fundus = load_img(img, target_size=input_size)

# https://stackoverflow.com/questions/43469281/how-to-predict-input-image-using-trained-model-in-keras
x = image.img_to_array(test_fundus)
x = np.expand_dims(x, axis=0)
images = np.vstack([x])
# end of stackoverflow

predicted = model.predict(images)

print("predicted shape: " + str(predicted.shape))
print("predicted type: "  + str(type(predicted)))

predicted.tofile('predicted.csv', sep=',')
print("Saved")

print("Done")
