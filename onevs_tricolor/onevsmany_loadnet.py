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

#input_size = (960, 1440, 3)
input_size = (160,160, 3)

smoothing_factor = float(input_size[0] * input_size[1])

def custom_loss(y_true, y_pred):
    # y_true: ground truth.  y_pred: predictions

    """
    y_true_real = tf.dtypes.cast(y_true, tf.float64)
    y_pred_real = tf.dtypes.cast(y_pred, tf.float64)

    # https://lars76.github.io/neural-networks/object-detection/losses-for-segmentation/
    def dice_loss(y_true, y_pred):
        numerator = 2 * tf.reduce_sum(y_true * y_pred, axis=(1,2,3))
        denominator = tf.reduce_sum(y_true + y_pred, axis=(1,2,3))

        return tf.reshape(1 - numerator / denominator, (-1, 1, 1))

    return binary_crossentropy(y_true_real, y_pred_real) + dice_loss(y_true_real, y_pred_real)
    """

    """
    smooth = 100
    intersection = tf.reduce_sum(K.abs(y_true_real * y_pred_real))
    sum_ = tf.reduce_sum(K.abs(y_true_real) + K.abs(y_pred_real))
    #intersection = K.sum(K.abs(y_true_real * y_pred_real), axis=-1)
    #sum_ = K.sum(K.abs(y_true_real) + K.abs(y_pred_real), axis=-1)
    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    return (1 - jac) * smooth
    """

    # weighted cross entropy
    # https://lars76.github.io/neural-networks/object-detection/losses-for-segmentation/
    def convert_to_logits(y_pred):
        # see https://github.com/tensorflow/tensorflow/blob/r1.10/tensorflow/python/keras/backend.py#L3525
        y_pred = tf.clip_by_value(y_pred, tf.keras.backend.epsilon(), 1 - tf.keras.backend.epsilon())
        return tf.log(y_pred / (1 - y_pred))

    # To decrease the number of false negatives, set beta>1. To decrease the number of false positives, set beta<1.
    beta = 0.9
    y_pred = convert_to_logits(y_pred)
    loss_ = tf.nn.weighted_cross_entropy_with_logits(logits=y_pred, targets=y_true, pos_weight=beta)
    # or reduce_sum and/or axis=-1
    return tf.reduce_mean(loss_)


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

## https://stackoverflow.com/questions/43469281/how-to-predict-input-image-using-trained-model-in-keras
#x = image.img_to_array(test_fundus)
#x = np.expand_dims(x, axis=0)
#images = np.vstack([x])
## end of stackoverflow

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

