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

if len(sys.argv) < 3:
    # There must be a command line argument with the class and run number to use.
    # First CL arg (element zero) is the name of this program.
    print('This program requires two command line arguments: run number (1..10000000), and class (0..2), in that order')
    sys.exit()

print("==================================")
print("arg 1: " + sys.argv[1])
print("arg 2: " + sys.argv[2])
print("==================================")

# Which class are we training this time?
class_active = int(sys.argv[2])
if class_active < 0 or class_active > 2:
    print('This program requires two command line arguments: run number (1..10000000), and class (0..2), in that order')
    sys.exit()

# We'll append this to the saved weights so we can run more than one
#   version of this code at once
run_number = int(sys.argv[1])
if run_number < 1 or run_number > 10000000:
    print('This program requires two command line arguments: run number (1..10000000), and class (0..2), in that order')
    sys.exit()

in_model_loc = 'best_model_incremental-cls'+ str(class_active) + '-run' + str(run_number) + '.h5'
if not os.path.isfile(in_model_loc):
    print('The file ' + best_model_loc + ' must exist in the directory with this executable.')
    sys.exit()


print("==================================")
print("run number: " + str(run_number))
print("class: " + str(class_active))
print("==================================")


num_fl = 130
num_fl_val = 32

path_fundus = '/worksite/fundus/'
path_fundus_val = '/worksite/fundusvalidate/'
path_mask = '/worksite/mask' + str(class_active) + '/'
path_mask_val = '/worksite/maskvalidate' + str(class_active) + '/'
out_path = '/outgoing/'
batchsize = 1
#input_size = (960, 1440, 3)
#input_size = (320, 480, 3)
input_size = (160, 160, 3)

neuron_default = 64

best_model_loc = out_path + 'best_model_incremental-cls'+ str(class_active) + '-run' + str(run_number) + '.h5'



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

    monitor_type = 'loss'
    loss = custom_loss

    #learning_rate = 1e-5
    #model.compile(optimizer = Adam(lr = learning_rate), loss = custom_loss, metrics = [soft_loss])

    model = tf.keras.models.load_model(in_model_loc,
           custom_objects={'soft_loss': soft_loss, 'custom_loss': custom_loss})
           
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
    best_model_loc = out_path + 'best_model_incremental-cls'+ str(class_active) + '-run' + str(run_number) + '.h5'
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
