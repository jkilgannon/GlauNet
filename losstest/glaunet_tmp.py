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
#from positional_loss import *


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
input_size = (320, 480, 3)

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


"""
def custom_loss(y_true, y_pred):
    y_true_real = np.array(tf.dtypes.cast(y_true, tf.float64))
    y_pred_real = np.array(tf.dtypes.cast(y_pred, tf.float64))

    xend = input_size[1] - 2          # -2 so we avoid the edges (which don't matter)
    yend = input_size[0] - 2          # -2 so we avoid the edges (which don't matter)
    numclasses = input_size[2]
    numerators = np.zeros((batchsize, numclasses), dtype=np.float64)

    # Calculate the local class possibilites.
    for batch in range(batchsize):
        for cls in range(numclasses):
            for x in range(2, xend):
                for y in range(2, yend):
                    # Doing it in numpy.  WAY faster than doing it in TensorFlow b/c of a
                    #  memory leak in TF.
                    numerators[batch, cls] = numerators[batch, cls] + abs(64 * y_true_real[batch, y, x, cls] - (30 * y_pred_real[batch, y, x, cls] + 3 * (y_pred_real[batch, y-1, x-1, cls] + y_pred_real[batch, y, x-1, cls] + y_pred_real[batch, y+1, x-1, cls] + y_pred_real[batch, y-1, x, cls] + y_pred_real[batch, y+1, x, cls] + y_pred_real[batch, y-1, x+1, cls] + y_pred_real[batch, y, x+1, cls] + y_pred_real[batch, y+1, x+1, cls]) + y_pred_real[batch, y-1, x-2, cls] + y_pred_real[batch, y, x-2, cls] + y_pred_real[batch, y+1, x-2, cls] + y_pred_real[batch, y-2, x-1, cls] + y_pred_real[batch, y+2, x-1, cls] + y_pred_real[batch, y-2, x, cls] + y_pred_real[batch, y+2, x, cls] + y_pred_real[batch, y-2, x+1, cls] + y_pred_real[batch, y+2, x+1, cls] + y_pred_real[batch, y-1, x+2, cls] + y_pred_real[batch, y, x+2, cls] + y_pred_real[batch, y+1, x+2, cls]))

    denominator = float(numclasses) * 66.0 * (input_size[0] - 4) * (input_size[1] - 4)

    # Calculate the loss for each batch.  We want to weight the classes
    #   equally, instead of letting one batch have too much influence.
    loss = [0] * batchsize
    for batch in range(batchsize):
        for cls in range(numclasses):
            loss[batch] = loss[batch] + numerators[batch, cls]
        loss[batch] = loss[batch] / denominator

    return tf.reshape(loss, (batchsize, 1, 1, 1))
"""

"""
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
"""
"""
def custom_loss(y_true, y_pred):
    #y_true = tf.reshape(y_true, (batchsize, input_size[0], input_size[1], input_size[2]))
    #y_pred = tf.reshape(y_pred, (batchsize, input_size[0], input_size[1], input_size[2]))

    #y_true_real = np.array(tf.dtypes.cast(keras.eval(y_true), tf.float64))
    #y_pred_real = np.array(tf.dtypes.cast(keras.eval(y_pred), tf.float64))

    y_true_real = keras.eval(tf.dtypes.cast(y_true, tf.float64))
    y_pred_real = keras.eval(tf.dtypes.cast(y_pred, tf.float64))

    #with tf.Session() as sess:
    #    y_true_real = np.array(tf.dtypes.cast(y_true.eval(session=sess), tf.float64))
    #    y_pred_real = np.array(tf.dtypes.cast(y_pred.eval(session=sess), tf.float64))


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
"""

def custom_loss(y_true, y_pred):
    # https://stackoverflow.com/questions/49192051/converting-tensor-to-np-array-using-k-eval-in-keras-returns-invalidargumenterr
    return keras.abs(keras.sum((y_true * 2 - keras.ones_like(y_true)) * y_pred))


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

            #print("image type: " + str(type(test_fundus)))
            #print("image shape: " + str(test_fundus.size))
            #vvvv = list(test_fundus.getdata())
            #print(vvvv[0])
            #print(vvvv[1])
            #print(vvvv[2])
            #print(vvvv[12000])
            ##print(str(test_fundus[0,0]) + " : " + str(test_fundus[0,1]))

            # https://stackoverflow.com/questions/43469281/how-to-predict-input-image-using-trained-model-in-keras
            x = image.img_to_array(test_fundus)

            #print("image array type: " + str(type(x)))
            #print("image array shape: " + str(x.shape))
            #print(str(x[0,0]) + " : " + str(x[0,1]))

            # Get the target data, which is a saved numpy ndarray
            y = np.load(target)

            #print("target type: " + str(type(y)))
            #print("target shape: " + str(y.shape))
            #print(str(y.shape[0]) + "," + str(y.shape[1]))
            #ccc = ""
            #for xxx in range(y.shape[0]):
            #    for yyy in range(y.shape[1]):
            #        ccc = ccc + str(y[xxx,yyy]) + ","
            #
            #text_file = open("image_array_as_text.txt", "w")
            #n = text_file.write(ccc)
            #text_file.close()
            #print("Done output")
            #print(str(y[0,0]) + " : " + str(y[0,1])  + " : " + str(y[0,2])  + " : " + str(y[0,3]))

            batch_head = batch_end + 1
            batch_end = batch_head + batchsize 

            #x_set = x.reshape((-1, 320, 480, 3))
            x_set = x.reshape((-1, input_size[0], input_size[1], input_size[2]))
            #y_shrunk = np.zeros((320,480), dtype=np.uint8)
            #y_shrunk = np.zeros((320,480,1), dtype=np.uint8)
            #y_shrunk = np.zeros((320,480,2), dtype=np.uint8)
            #y_shrunk = np.zeros((320,480,3), dtype=np.uint8)
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
            y_set = y_shrunk.reshape((-1, input_size[0], input_size[1], input_size[2]))

            #print("x_set array shape: " + str(x_set.shape))
            #print("y_set array shape: " + str(y_set.shape))
            #print(x_set.shape) print(y_set.shape) print(y_set) print("--") print(str(counter)) counter += 1

            yield (x_set, y_set)


def main():
    keras.clear_session()

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
    #conv10 = Conv2D(2, 1, activation = 'sigmoid')(conv9)
    #conv10 = Conv2D(1, 1, activation = 'sigmoid')(conv9)
    conv10 = Conv2D(3, 1, activation = 'sigmoid')(conv9)

    model = Model(inputs = input_layer, outputs = conv10)

    #loss_type = 'binary_crossentropy'
    monitor_type = 'loss'

    #model.compile(optimizer = Adam(lr = 1e-4), loss = loss_type, metrics = ['accuracy'])
    #model.compile(optimizer = Adam(lr = 1e-4), loss = 'cosine_similarity', metrics = ['accuracy'])

    #model.compile(optimizer = Adam(lr = 1e-4), loss = iou_coeff, metrics = ['accuracy'])
    #model.compile(optimizer = Adam(lr = 1e-4), loss = dice_coeff_inverted, metrics = [dice_coeff])
    #model.compile(optimizer = Adam(lr = 1e-4), loss = dice_loss_2, metrics = [inverted_dice_2])
    model.compile(optimizer = Adam(lr = 1e-4), loss = custom_loss, metrics = [soft_loss])

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
    best_model_loc = out_path + 'best_model_incremental_' + str(run_number) + '.h5'
    CHKPT = ModelCheckpoint(best_model_loc, 
                         monitor=monitor_type, 
                         mode='min', 
                         verbose=1, 
                         save_best_only=True)


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
