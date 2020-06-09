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
from tensorflow.keras.losses import binary_crossentropy

# Previous version: GlauNet/onevs/onevsmany_loadnet.py 

#input_size = (960, 1440, 3)
input_size = (160,160, 3)

smoothing_factor = float(input_size[0] * input_size[1])

def custom_loss(y_true, y_pred):
    # y_true: ground truth.  y_pred: predictions
    #
    # This version uses the Jaccard loss (IOU) described in https://arxiv.org/pdf/1801.05746.pdf

    #y_true_real = tf.dtypes.cast(y_true, tf.float64)
    #y_pred_real = tf.dtypes.cast(y_pred, tf.float64)

    # y_true is 1 for correct pixels, and 0 for incorrect ones

    # https://github.com/keras-team/keras-contrib/blob/master/keras_contrib/losses/jaccard.py
    # Jaccard = (|X & Y|)/ (|X|+ |Y| - |X & Y|)
    #         = sum(|A*B|)/(sum(|A|)+sum(|B|)-sum(|A*B|))
    y_true = tf.reshape(y_true, (-1, input_size[0], input_size[1], 1))
    smooth = 100
    intersection = tf.reduce_sum(K.abs(y_true * y_pred))
    sum_ = tf.reduce_sum(K.abs(y_true) + K.abs(y_pred))
    #intersection = K.sum(K.abs(y_true_real * y_pred_real), axis=-1)
    #sum_ = K.sum(K.abs(y_true_real) + K.abs(y_pred_real), axis=-1)
    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    return (1 - jac) * smooth

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
x = np.reshape(x, input_size)
images = np.reshape(x, (-1, input_size[0], input_size[1], input_size[2]))
# end of stackoverflow

# Output the input file to make sure it's good
images.tofile('input_image_readable.csv', sep=',')

predicted = model.predict(images)

print("predicted shape: " + str(predicted.shape))
print("predicted type: "  + str(type(predicted)))

predicted.tofile('predicted.csv', sep=',')
print("Saved")

print("Done")
