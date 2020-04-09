from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from tensorflow.keras import backend as keras
import os
import numpy as np
import tensorflow as tf
from PIL import Image
from random import randint


#batchsize = 3
batchsize = 1
#input_size = (960, 1440, 3)
input_size = (320, 480, 3)

#sess = tf.Session()

smoothing_factor = float(input_size[0] * input_size[1])
print("smoothing factor: " + str(smoothing_factor))

#y_true = tf.Variable(tf.zeros([batchsize, input_size[0], input_size[1], input_size[2]]))
#y_pred = tf.Variable(tf.zeros([batchsize, input_size[0], input_size[1], input_size[2]]))
y_true = tf.placeholder(tf.float64, shape=(batchsize, input_size[0], input_size[1], input_size[2]))
y_pred = tf.placeholder(tf.float64, shape=(batchsize, input_size[0], input_size[1], input_size[2]))


def custom_loss(y_true, y_pred):
    y_true = tf.reshape(y_true, (batchsize, input_size[0], input_size[1], input_size[2]))
    y_pred = tf.reshape(y_pred, (batchsize, input_size[0], input_size[1], input_size[2]))

    y_true_real = np.array(tf.dtypes.cast(keras.eval(y_true), tf.float64))
    y_pred_real = np.array(tf.dtypes.cast(keras.eval(y_pred), tf.float64))

    #y_true_real = y_true_real.reshape(batchsize, input_size[0], input_size[1], input_size[2])
    #y_pred_real = y_pred_real.reshape(batchsize, input_size[0], input_size[1], input_size[2])

    print("=_=_=_=_=")
    print(y_true.get_shape().as_list())
    print(y_true_real.shape)
    print("=_=_=_=_=")

    xend = input_size[1] - 2          # -2 so we avoid the edges (which don't matter)
    yend = input_size[0] - 2          # -2 so we avoid the edges (which don't matter)
    numclasses = input_size[2]
    numerators = np.zeros((numclasses), dtype=np.float64)

    # Calculate the local class possibilites.
    for batch in range(batchsize):
        for cls in range(numclasses):
            for x in range(2, xend):
                for y in range(2, yend):
                    # Doing it in numpy.  WAY faster than doing it in TensorFlow b/c of a
                    #  memory leak in TF.
                    numerators[batch, cls] = numerators[batch, cls] + abs(66.0 * y_true[batch, y, x, cls] - (30 * y_pred[batch, y, x, cls] + 3.0 * (y_pred[batch, y-1, x-1, cls] + y_pred[batch, y, x-1, cls] + y_pred[batch, y+1, x-1, cls] + y_pred[batch, y-1, x, cls] + y_pred[batch, y+1, x, cls] + y_pred[batch, y-1, x+1, cls] + y_pred[batch, y, x+1, cls] + y_pred[batch, y+1, x+1, cls]) + y_pred[batch, y-1, x-2, cls] + y_pred[batch, y, x-2, cls] + y_pred[batch, y+1, x-2, cls] + y_pred[batch, y-2, x-1, cls] + y_pred[batch, y+2, x-1, cls] + y_pred[batch, y-2, x, cls] + y_pred[batch, y+2, x, cls] + y_pred[batch, y-2, x+1, cls] + y_pred[batch, y+2, x+1, cls] + y_pred[batch, y-1, x+2, cls] + y_pred[batch, y, x+2, cls] + y_pred[batch, y+1, x+2, cls]))

    denominator = float(numclasses) * 66.0 * (input_size[0] - 4) * (input_size[1] - 4)

    # Calculate the loss for each batch.  We want to weight the classes
    #   equally, instead of letting one batch have too much influence.
    loss = [0] * batchsize
    for batch in range(batchsize):
        for cls in range(numclasses):
            loss[batch] = loss[batch] + numerators[batch, cls]
        loss[batch] = loss[batch] / denominator

    return tf.reshape(loss, (batchsize, 1, 1, 1))


def soft_loss(y_true, y_pred):
    return 1 - custom_loss(y_true, y_pred)


