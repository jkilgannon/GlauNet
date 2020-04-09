from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from tensorflow.keras.preprocessing import image

from tensorflow.keras.preprocessing.image import load_img

from tensorflow.keras.preprocessing.image import img_to_array

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.models import *

from tensorflow.keras.layers import *

from tensorflow.keras.optimizers import *

from tensorflow.keras import backend as keras

import os

import numpy as np

import tensorflow as tf

from PIL import Image

import matplotlib.pyplot as plt

input_size = (320, 480, 3)


smoothing_factor = float(input_size[0] * input_size[1])

print("smoothing factor: " + str(smoothing_factor))


# Original idea: https://lars76.github.io/neural-networks/object-detection/losses-for-segmentation/

# Casting to float, and using axis(), makes this work (original didn't cast)

# Also needed to add the smooth, otherwise this could return nan (div by zero)

def dice_coeff(y_true, y_pred):

    # The smoothing factor is the number of pixels in one layer

    #   of the image.  This is so that we scale up the coeff, so

    #   there is more "space" for the code to search in.  If 

    #   smooth is near 1, the loss stays near 1 until we have nearly

    #   the entire image correctly calculated!


    y_true_real = tf.dtypes.cast(y_true, tf.float32)

    y_pred_real = tf.dtypes.cast(y_pred, tf.float32)

    

    numerators = 2 * tf.reduce_sum(y_true_real * y_pred_real, axis=(1,2))

    denominators = tf.reduce_sum(y_true_real + y_pred_real, axis=(1,2))

    

    print(";;;;;;;")

    print(numerators)

    print(denominators)

    print(";;;;;;;")

    

    numerator = smoothing_factor + tf.reduce_mean(numerators)

    denominator = smoothing_factor + tf.reduce_mean(denominators)


    #numerator = smoothing_factor + (2 * tf.reduce_sum(y_true_real * y_pred_real, axis=(1,2,3)))

    #denominator = smoothing_factor + tf.reduce_sum(y_true_real + y_pred_real, axis=(1,2,3))

    

    return tf.reshape(numerator / denominator, (-1, 1, 1, 1))


def soft_dice_loss(y_true, y_pred):

    return 1 - dice_coeff(y_true, y_pred)


    	

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

model = load_model(prevmodelfile, custom_objects={'dice_coeff': dice_coeff, 'soft_dice_loss': soft_dice_loss})

print("++++++++++++++")

print(model.count_params())

print("++++++++++++++")

print(model.summary())

print("++++++++++++++")


img = 'test.tif'

test_fundus = load_img(img, target_size=input_size)


# https://stackoverflow.com/questions/43469281/how-to-predict-input-image-using-trained-model-in-keras

x = image.img_to_array(test_fundus)

x = np.expand_dims(x, axis=0)

images = np.vstack([x])

# end of stackoverflow


#test_fundus = Image.open(img)

predicted = model.predict(images)


print("predicted shape: " + str(predicted.shape))

print("predicted type: "  + str(type(predicted)))


predicted.tofile('predicted.csv', sep=',')

print("Saved")


print("Done")
