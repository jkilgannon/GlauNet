from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
#from tensorflow.keras import backend as keras
from tensorflow.keras import backend as K
import os
import numpy as np
import tensorflow as tf
from PIL import Image
from random import randint
#from positional_loss import *
import math
from tensorflow.keras.losses import BinaryCrossentropy


num_fl = 130
num_fl_val = 32

path_fundus = '/worksite/fundus/'
path_fundus_val = '/worksite/fundusvalidate/'
path_mask = '/worksite/mask/'
path_mask_val = '/worksite/maskvalidate/'
out_path = '/outgoing/'
#batchsize = 3
batchsize = 1
#input_size = (960, 1440, 3)
#input_size = (320, 480, 3)
input_size = (160, 160, 3)

#sess = tf.Session()

smoothing_factor = float(input_size[0] * input_size[1])
print("smoothing factor: " + str(smoothing_factor))

#y_true = tf.Variable(tf.zeros([batchsize, input_size[0], input_size[1], input_size[2]]))
#y_pred = tf.Variable(tf.zeros([batchsize, input_size[0], input_size[1], input_size[2]]))
#y_true = tf.placeholder(tf.float64, shape=(batchsize, input_size[0], input_size[1], input_size[2]))
#y_pred = tf.placeholder(tf.float64, shape=(batchsize, input_size[0], input_size[1], input_size[2]))


"""
# Set up a test case with all '1' that can be used to force the
#  network to keep away from the local optimum in which one
#  class is all '1'
all_ones = [1.0] * (input_size[0] * input_size[1])
all_ones_class = np.array(all_ones)
all_zeroes_class = np.zeros(input_size[0], input_size[1])), dtype=np.uint8)
"""

# We'll append this to the saved weights so we can run more than one
#   version of this code at once
run_number = randint(1, 10000000)

neuron_default = 64

#####################################

"""
def custom_loss(y_true, y_pred):
    # Per https://arxiv.org/pdf/1801.05746.pdf

    H = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y_true, logits=y_pred))

    #sess = tf.Session()

    tensor_J = tf.reduce_sum(jaccard_loss(y_true, y_pred))

    with tf.compat.v1.Session() as sess:
        scalar_J = sess.run(tensor_J.eval())
    #scalar_J = K.eval(tf.reduce_sum(jaccard_loss(y_true, y_pred)))
    #scalar_J = tf.reduce_sum(jaccard_loss(y_true, y_pred)).eval(session=sess)

    #scalar_J = tf.reshape(jaccard_loss(y_true, y_pred), []).numpy()
    logJ = math.log(scalar_J)
    return H - logJ

    #return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y_true, logits=y_pred)) - math.log(jaccard_loss(y_true, y_pred))
    #return BinaryCrossentropy(y_true, y_pred) - math.log(jaccard_loss(y_true, y_pred))
    #return BinaryCrossentropy(y_true, y_pred) - math.log(jaccard_loss(y_true, y_pred))
"""

#def jaccard_loss(y_true, y_pred):
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
    #counter = 0

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
            target = annotated_loc + files[batch_head]

            test_fundus = load_img(img, target_size=input_size)

            # https://stackoverflow.com/questions/43469281/how-to-predict-input-image-using-trained-model-in-keras
            x = image.img_to_array(test_fundus)

            # Get the target data, which is a saved numpy ndarray
            y = np.load(target)

            ## Used for tricolor test
            #y = np.fromfile(target)  # Used for tricolor test
            #y = y.reshape((input_size[0], input_size[1]))
            ## Used for tricolor test

            batch_head = batch_end + 1
            batch_end = batch_head + batchsize

            #x_set = x.reshape((-1, 320, 480, 3))
            x_set = x.reshape((-1, input_size[0], input_size[1], input_size[2]))

            """
            y_shrunk = np.zeros((input_size[0], input_size[1], input_size[2]), dtype=np.double)

            for row_small in range(0, input_size[0]):
                for col_small in range(0, input_size[1]):
                    row = row_small * 3
                    col = col_small * 3
                    total_value = y[row,col] + y[row,col+1] + y[row,col+2] + y[row+1,col] + y[row+1,col+1] + y[row+1,col+2] + y[row+2,col] + y[row+2,col+1] + y[row+2,col+2]

                    # One-hot the categories
                    avg_val = round(total_value / 9.0)
                    if avg_val == 0:
                        y_shrunk[row_small,col_small,0] = 1.0
                    elif avg_val == 1:
                        y_shrunk[row_small,col_small,1] = 1.0
                    else:
                        y_shrunk[row_small,col_small,2] = 1.0

            #print("shrunk array shape: " + str(y_shrunk.shape))

            #y_set = np.array(y, dtype=np.uint8)
            #y_set = y_shrunk.reshape((-1, 320, 480, 3))
            #y_shrunk = np.zeros((320,480), dtype=np.uint8)
            #y_shrunk = np.zeros((320,480,1), dtype=np.uint8)
            #y_shrunk = np.zeros((320,480,2), dtype=np.uint8)
            #y_shrunk = np.zeros((320,480,3), dtype=np.uint8)
            """

            y_shrunk = np.zeros((input_size[0], input_size[1], input_size[2]), dtype=np.double)

            for row in range(0, input_size[0]):
                for col in range(0, input_size[1]):
                    # One-hot the categories
                    if y[row, col] == 0:
                        y_shrunk[row, col, 0] = 1.0
                    elif y[row, col] == 1:
                        y_shrunk[row, col, 1] = 1.0
                    else:
                        y_shrunk[row, col, 2] = 1.0

            y_set = y_shrunk.reshape((-1, input_size[0], input_size[1], input_size[2]))

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
    #merge6 = concatenate([conv4,up6])
    #drop6 = Dropout(0.25)(merge6)
    drop6 = Dropout(0.25)(up6)
    conv6 = Conv2D(neuron_default * 8, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(drop6)
    conv6 = Conv2D(neuron_default * 8, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)

    up7 = Conv2DTranspose(neuron_default * 4, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6))
    #merge7 = concatenate([conv3,up7])
    #drop7 = Dropout(0.25)(merge7)
    drop7 = Dropout(0.25)(up7)
    conv7 = Conv2D(neuron_default * 4, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(drop7)
    conv7 = Conv2D(neuron_default * 4, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)

    up8 = Conv2DTranspose(neuron_default * 2, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7))
    #merge8 = concatenate([conv2,up8])
    #drop8 = Dropout(0.25)(merge8)
    drop8 = Dropout(0.25)(up8)
    conv8 = Conv2D(neuron_default * 2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(drop8)
    conv8 = Conv2D(neuron_default * 2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)

    up9 = Conv2DTranspose(neuron_default, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8))
    #merge9 = concatenate([conv1,up9])
    #drop9 = Dropout(0.25)(merge9)
    drop9 = Dropout(0.25)(up9)
    conv9 = Conv2D(neuron_default, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(drop9)
    conv9 = Conv2D(neuron_default, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)

    # Input shape: 4D tensor with shape: (samples, channels, rows, cols)
    # Output shape: 4D tensor with shape: (samples, filters, new_rows, new_cols) 
    #   if data_format='channels_first' or 4D tensor with shape: (samples, new_rows, new_cols, filters) 
    #   if data_format='channels_last'. rows and cols values might have changed due to padding.
    #conv10 = Conv2D(2, 1, activation = 'sigmoid')(conv9)
    #conv10 = Conv2D(1, 1, activation = 'sigmoid')(conv9)
    conv10 = Conv2D(3, 1, activation = 'softmax')(conv9)

    model = Model(inputs = input_layer, outputs = conv10)

    #loss_type = 'binary_crossentropy'
    monitor_type = 'loss'

    #model.compile(optimizer = Adam(lr = 1e-4), loss = loss_type, metrics = ['accuracy'])
    #model.compile(optimizer = Adam(lr = 1e-4), loss = 'cosine_similarity', metrics = ['accuracy'])

    #model.compile(optimizer = Adam(lr = 1e-4), loss = iou_coeff, metrics = ['accuracy'])
    #model.compile(optimizer = Adam(lr = 1e-4), loss = dice_coeff_inverted, metrics = [dice_coeff])
    #model.compile(optimizer = Adam(lr = 1e-4), loss = dice_loss_2, metrics = [inverted_dice_2])
    #model.compile(optimizer = Adam(lr = 1e-4), loss = custom_loss, metrics = [soft_loss])
    model.compile(optimizer = Adam(lr = 0.001), loss = custom_loss, metrics = [soft_loss])

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
    best_model_loc = out_path + 'best_model_incremental_' + str(run_number) + '.h5'
    CHKPT = ModelCheckpoint(best_model_loc, 
                         monitor=monitor_type, 
                         mode='min', 
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
                        shuffle=True,
                        epochs=500,
                        validation_data=batch_gen_val,
                        validation_steps=num_fl_val // batchsize,
                        callbacks=[EARLYSTOP, CHKPT])

    # Save the final model
    final_model_loc = out_path + 'last_weights_' + str(run_number) + '.h5'
    model.save(final_model_loc)

    print("Done")


# Launch!
main()

