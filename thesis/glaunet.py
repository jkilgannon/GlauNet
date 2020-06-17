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

# Previous version(s): nn/onevsmany_LR_WORKING_glaunet.py

#num_fl = 130
#num_fl_val = 32

# Which class are we training this time?
class_active = 1

path_fundus = '/worksite/fundus/'
path_fundus_val = '/worksite/fundusvalidate/'
path_mask = '/worksite/mask' + str(class_active) + '/'
path_mask_val = '/worksite/maskvalidate' + str(class_active) + '/'
out_path = '/outgoing/'
#batchsize = 3
batchsize = 1
#input_size = (960, 1440, 3)
#input_size = (320, 480, 3)
input_size = (160, 160, 3)

# Get the number of files in the training images (num_fl) and the
# validation images (num_fl_val).
batchpath, batchdirs, batchfiles = next(os.walk(path_fundus))
num_fl = len(batchfiles)
batchpath, batchdirs, batchfiles = next(os.walk(path_fundus_val))
num_fl_val = len(batchfiles)

# now check that there are the same number of ground truth and mask images
batchpath, batchdirs, batchfiles = next(os.walk(path_mask))
num_fl_mask = len(batchfiles)
batchpath, batchdirs, batchfiles = next(os.walk(path_mask_val))
num_fl_val_mask = len(batchfiles)

if num_fl != num_fl_mask or num_fl_val != num_fl_val_mask:
   print("Number of ground truth images does not match number of mask images. Exiting.")
   sys.exit()

print("Training/validation image counts")
print(num_fl)
print(num_fl_val)

# We'll append this to the saved weights so we can run more than one
#   version of this code at once
run_number = randint(1, 10000000)
print("==================================")
print("run number: " + str(run_number))
print("==================================")

neuron_default = 64

#####################################

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


def main():
    K.clear_session()

    #sess.run(tf.global_variables_initializer())

    # UNet:
    # https://github.com/zhixuhao/unet

    input_layer = Input(input_size)

    conv1 = Conv2D(neuron_default, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(input_layer)
    conv1 = Conv2D(neuron_default, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    drop1 = Dropout(0.25)(pool1)

    conv2 = Conv2D(neuron_default * 2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(drop1)
    conv2 = Conv2D(neuron_default * 2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    drop2 = Dropout(0.25)(pool2)

    conv3 = Conv2D(neuron_default * 4, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(drop2)
    conv3 = Conv2D(neuron_default * 4, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    drop3 = Dropout(0.25)(pool3)

    conv4 = Conv2D(neuron_default * 8, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(drop3)
    conv4 = Conv2D(neuron_default * 8, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
    drop4 = Dropout(0.25)(pool4)

    ## Center point
    conv5 = Conv2D(neuron_default * 16, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(drop4)
    conv5 = Conv2D(neuron_default * 16, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
    drop5 = Dropout(0.5)(conv5)

    up6 = UpSampling2D(size = (2,2))(drop5)
    up6 = Conv2DTranspose(neuron_default * 8, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(up6)
    merge6 = concatenate([conv4,up6])
    drop6 = Dropout(0.25)(merge6)
    conv6 = Conv2D(neuron_default * 8, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(drop6)
    conv6 = Conv2D(neuron_default * 8, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)

    up7 = Conv2DTranspose(neuron_default * 4, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6))
    merge7 = concatenate([conv3,up7])
    drop7 = Dropout(0.25)(merge7)
    conv7 = Conv2D(neuron_default * 4, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(drop7)
    conv7 = Conv2D(neuron_default * 4, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)

    up8 = Conv2DTranspose(neuron_default * 2, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7))
    merge8 = concatenate([conv2,up8])
    drop8 = Dropout(0.25)(merge8)
    conv8 = Conv2D(neuron_default * 2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(drop8)
    conv8 = Conv2D(neuron_default * 2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)

    up9 = Conv2DTranspose(neuron_default, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8))
    merge9 = concatenate([conv1,up9])
    drop9 = Dropout(0.25)(merge9)
    conv9 = Conv2D(neuron_default, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(drop9)
    conv9 = Conv2D(neuron_default, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)

    # Input shape: 4D tensor with shape: (samples, channels, rows, cols)
    # Output shape: 4D tensor with shape: (samples, filters, new_rows, new_cols) 
    #   if data_format='channels_first' or 4D tensor with shape: (samples, new_rows, new_cols, filters) 
    #   if data_format='channels_last'. rows and cols values might have changed due to padding.
    conv10 = Conv2D(1, 1, activation = 'sigmoid')(conv9)

    model = Model(inputs = input_layer, outputs = conv10)

    #loss_type = 'binary_crossentropy'
    monitor_type = 'loss'

    #model.compile(optimizer = Adam(lr = 1e-4), loss = loss_type, metrics = ['accuracy'])
    #model.compile(optimizer = Adam(lr = 1e-4), loss = 'cosine_similarity', metrics = ['accuracy'])
    #model.compile(optimizer = Adam(lr = 1e-4), loss = iou_coeff, metrics = ['accuracy'])
    #model.compile(optimizer = Adam(lr = 1e-4), loss = dice_coeff_inverted, metrics = [dice_coeff])
    #model.compile(optimizer = Adam(lr = 1e-4), loss = dice_loss_2, metrics = [inverted_dice_2])

    learning_rate = 1e-5
    model.compile(optimizer = Adam(lr = learning_rate), loss = custom_loss, metrics = [soft_loss])

    print("++++++++++++++")
    print(model.count_params())
    print("++++++++++++++")
    print(model.summary())
    print("++++++++++++++")
    os.system('free -m')
    print("++++++++++++++")
    os.system('vmstat -s')
    print("++++++++++++++")


    EARLYSTOP = EarlyStopping(patience=50, 
                              monitor=monitor_type, 
                              restore_best_weights=True)

    # Save off the very best model we can find; avoids overfitting.
    #best_model_loc = out_path + 'best_model_incremental_{epoch:02d}_' + str(run_number) + '.h5'
    best_model_loc = out_path + 'best_model_incremental-cls'+ str(class_active) + '-run' + str(run_number) + '.h5'
    #CHKPT = ModelCheckpoint(best_model_loc, 
    #                     monitor=monitor_type, 
    #                     mode='min', 
    #                     verbose=1)
    CHKPT = ModelCheckpoint(best_model_loc, 
                         monitor=monitor_type, 
                         mode='min', 
                         save_weights_only = False,
                         verbose=1)

    # Set up the batch generator
    batch_gen = batchmaker(path_fundus, 
                       path_mask, 
                       batchsize, input_size)

    batch_gen_val = batchmaker(path_fundus_val, 
                       path_mask_val, 
                       batchsize, input_size)

    history = model.fit_generator(batch_gen,
                        steps_per_epoch=num_fl // batchsize,
                        shuffle=False,
                        epochs=500,
                        validation_data=batch_gen_val,
                        validation_steps=num_fl_val // batchsize,
                        callbacks=[EARLYSTOP, CHKPT])

    # Save the final model
    #final_model_loc = out_path + 'last_weights_-cls'+ str(class_active) + '-run' + str(run_number) + '.h5'
    #model.save(final_model_loc)
    final_model_loc = out_path + 'final_model_-cls'+ str(class_active) + '-run' + str(run_number) + '.tf'
    model.save(final_model_loc, save_format='tf')

    print("Done")


# Launch!
main()
