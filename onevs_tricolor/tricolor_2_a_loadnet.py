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


#input_size = (960, 1440, 3)
input_size = (160,160, 3)

smoothing_factor = float(input_size[0] * input_size[1])

def custom_loss(y_true, y_pred):
    # y_true: ground truth.  y_pred: predictions
    #
    # We will calculate the loss for each class separately and then take the mean of them
    # so that we don't have the largest class dominate the loss calculations.  (The smaller
    # ones will have much more weight than usual, which is expected from our data, where
    # the cup and disc are tiny compared to the background.)

    # This version uses the Jaccard loss (IOU) described in https://arxiv.org/pdf/1801.05746.pdf

    y_true_real = tf.dtypes.cast(y_true, tf.float64)
    y_pred_real = tf.dtypes.cast(y_pred, tf.float64)

    # Separate the ground truth and prediction into their classes.
    y_true_cls0 = y_true_real[:,:,:,0]
    y_true_cls1 = y_true_real[:,:,:,1]
    y_true_cls2 = y_true_real[:,:,:,2]

    y_pred_cls0 = y_pred_real[:,:,:,0]
    y_pred_cls1 = y_pred_real[:,:,:,1]
    y_pred_cls2 = y_pred_real[:,:,:,2]

    # y_true is 1 for correct pixels, and 0 for incorrect ones

    # Jaccard/IOU by class:
    J_0 = (1/smoothing_factor) * tf.reduce_sum((y_true_cls0 * y_pred_cls0) / (K.epsilon() + y_pred_cls0 + y_true_cls0 - (y_true_cls0 * y_pred_cls0)))
    J_1 = (1/smoothing_factor) * tf.reduce_sum((y_true_cls1 * y_pred_cls1) / (K.epsilon() + y_pred_cls1 + y_true_cls1 - (y_true_cls1 * y_pred_cls1)))
    J_2 = (1/smoothing_factor) * tf.reduce_sum((y_true_cls2 * y_pred_cls2) / (K.epsilon() + y_pred_cls2 + y_true_cls2 - (y_true_cls2 * y_pred_cls2)))

    """
    # Dice coefficient for each of the three classes.
    dice_0 = (2 * tf.reduce_sum(y_true_cls0 * y_pred_cls0)) / (tf.reduce_sum(K.square(y_true_cls0)) + tf.reduce_sum(K.square(y_pred_cls0)) + K.epsilon())
    dice_1 = (2 * tf.reduce_sum(y_true_cls1 * y_pred_cls1)) / (tf.reduce_sum(K.square(y_true_cls1)) + tf.reduce_sum(K.square(y_pred_cls1)) + K.epsilon())
    dice_2 = (2 * tf.reduce_sum(y_true_cls2 * y_pred_cls2)) / (tf.reduce_sum(K.square(y_true_cls2)) + tf.reduce_sum(K.square(y_pred_cls2)) + K.epsilon())

    #dice_0 = (2 * tf.reduce_sum(y_true_cls0 * y_pred_cls0)) / (tf.reduce_sum(y_true_cls0 * y_true_cls0) + tf.reduce_sum(y_pred_cls0 * y_pred_cls0) + K.epsilon())
    #dice_1 = (2 * tf.reduce_sum(y_true_cls1 * y_pred_cls1)) / (tf.reduce_sum(y_true_cls1 * y_true_cls1) + tf.reduce_sum(y_pred_cls1 * y_pred_cls1) + K.epsilon())
    #dice_2 = (2 * tf.reduce_sum(y_true_cls2 * y_pred_cls2)) / (tf.reduce_sum(y_true_cls2 * y_true_cls2) + tf.reduce_sum(y_pred_cls2 * y_pred_cls2) + K.epsilon())

    return tf.reshape(tf.reduce_mean(dice_0 + dice_1 + dice_2) / 3.0, (-1,1,1,1))
    """

    H = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y_true, logits=y_pred))
    H = tf.dtypes.cast(H, tf.float64)

    loss_total = H - tf.math.log((J_0 + J_1 + J_2) / 3.0)

    return tf.reshape(loss_total, (-1,1,1,1))

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

