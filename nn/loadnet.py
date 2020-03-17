from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
#from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from tensorflow.keras import backend as K
#from tensorflow.keras import backend as keras
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


#def custom_loss(y_true, y_pred):
#    # https://stackoverflow.com/questions/49192051/converting-tensor-to-np-array-using-k-eval-in-keras-returns-invalidargumenterr
#    return keras.abs(keras.sum((y_true * 2 - keras.ones_like(y_true)) * y_pred))


#def soft_loss(y_true, y_pred):
#    return 1 - custom_loss(y_true, y_pred)
"""
def custom_loss(y_true, y_pred):
    # Dice loss.
    # https://stackoverflow.com/questions/51100508/implementing-custom-loss-function-in-keras-with-condition
    #y_pred = y_pred > thresh
    smooth = 1e-5
    thresh = 0.5

    y_true_f = keras.flatten(y_true)
    y_pred_f = keras.flatten(y_pred)
    intersection = keras.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (keras.sum(y_true_f) + keras.sum(y_pred_f) + smooth)

    ## https://stackoverflow.com/questions/49192051/converting-tensor-to-np-array-using-k-eval-in-keras-returns-invalidargumenterr
    #return keras.abs(keras.sum((y_true * 2 - keras.ones_like(y_true)) * y_pred))


#def soft_loss(y_true, y_pred):
#    return 1 - custom_loss(y_true, y_pred)

def soft_loss(y_true, y_pred):
    return -custom_loss(y_true, y_pred)
"""

def custom_loss(y_true, y_pred):
    # y_true: ground truth.  y_pred: predictions
    #
    # The logic behind this is that there are two factors in correctness:
    # 1) How correct is the prediction?  This is calculated by:
    #       A = prediction * ground truth.
    #    This gives a "mask" which zeroes out all the incorrect pixels in
    #    the prediction, leaving only the values [0..1] that it predicted
    #    for pixels that are in the ground truth mask.
    #    We then take |A|, the sum of all the predictions [0..1] within the
    #    corrrect mask.
    #       |GT| = Total number of pixels in the correct (ground truth) mask
    #    Factor alpha, then, is |A| / |GT|.
    #    Factor alpha is 0 when the prediction is awful, and 1 when it is perfect.
    #
    # 2) How incorrect is the prediction?  This is calculated by:
    #       |B| = |prediction| - |A|
    #    B would be the mask of all pixels that aren't in the ground truth.
    #    We don't need to calculate the mask; we just calculate the sum of
    #    all incorrect predictions.
    #    Factor beta, then, is |B| / |predictions|.
    #    Factor beta is 0 when the prediction is perfect, and 1 when it is awful.
    #
    # Now we can calculate the loss:
    #    loss = [abs(alpha - 1) + beta] / 2
    #
    # We will calculate the loss for each class separately and then take the mean of them
    # so that we don't have the largest class dominate the loss calculations.  (The smaller
    # ones will have much more weight than usual, which is expected from our data, where
    # the cup and disc are tiny compared to the background.)

    y_true_real = tf.dtypes.cast(y_true, tf.float64)
    y_pred_real = tf.dtypes.cast(y_pred, tf.float64)

    # Separate the ground truth and prediction into their classes.
    y_true_cls0 = y_true_real[:,:,:,0]
    y_true_cls1 = y_true_real[:,:,:,1]
    y_true_cls2 = y_true_real[:,:,:,2]

    y_pred_cls0 = y_pred_real[:,:,:,0]
    y_pred_cls1 = y_pred_real[:,:,:,1]
    y_pred_cls2 = y_pred_real[:,:,:,2]

    # y_true is 1 for correct pixels, and 0 for incorrect ones, so positive_diff is
    # a "mask" of the right values in y_pred.  And negative_diff is a "mask" of the wrong
    # values in y_pred.  The size variables contain the sums of those masks.

    # Calculate A, GT, and factor alpha.
    GT_size_0 = tf.reduce_sum(y_true_cls0, axis=(1,2))
    A_size_0 = tf.reduce_sum((y_true_cls0 * y_pred_cls0), axis=(1,2))
    alpha_0 = A_size_0 / GT_size_0

    GT_size_1 = tf.reduce_sum(y_true_cls1, axis=(1,2))
    A_size_1 = tf.reduce_sum((y_true_cls1 * y_pred_cls1), axis=(1,2))
    alpha_1 = A_size_1 / GT_size_1

    GT_size_2 = tf.reduce_sum(y_true_cls2, axis=(1,2))
    A_size_2 = tf.reduce_sum((y_true_cls2 * y_pred_cls2), axis=(1,2))
    alpha_2 = A_size_2 / GT_size_2

    #positive_diff = y_true_cls0 * y_pred_cls0
    #B_size_0 = A_size_0 - tf.reduce_sum(positive_diff, axis=(1,2))

    # Calculate B, prediction, and factor beta.
    pred_size_0 = tf.reduce_sum(y_pred_cls0, axis=(1,2))
    B_size_0 = pred_size_0 - A_size_0
    beta_0 = B_size_0 / pred_size_0

    pred_size_1 = tf.reduce_sum(y_pred_cls1, axis=(1,2))
    B_size_1 = pred_size_1 - A_size_1
    beta_1 = B_size_1 / pred_size_1

    pred_size_2 = tf.reduce_sum(y_pred_cls2, axis=(1,2))
    B_size_2 = pred_size_2 - A_size_2
    beta_2 = B_size_2 / pred_size_2

    # Calculate final losses
    loss_0 = (K.abs(1 - alpha_0) + beta_0) / 2.0
    loss_1 = (K.abs(1 - alpha_1) + beta_1) / 2.0
    loss_2 = (K.abs(1 - alpha_2) + beta_2) / 2.0
    #loss = (loss_0 + loss_1 + loss_2) / 3.0
    return tf.reshape(tf.reduce_sum(loss_0 + loss_1 + loss_2) / 3.0, (-1,1,1,1))


def soft_loss(y_true, y_pred):
    return 1 - custom_loss(y_true, y_pred)



    	
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
model = load_model(prevmodelfile, custom_objects={'soft_loss': soft_loss, 'custom_loss': custom_loss})
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
