from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
#from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from tensorflow.keras import backend as keras
import os
import numpy as np
import tensorflow as tf
from PIL import Image


#https://github.com/keras-team/keras/issues/5720
def combine_generator(gen1, gen2):
    while True:
        yield(gen1.next(), gen2.next())


# https://stackoverflow.com/questions/53248099/keras-image-segmentation-using-grayscale-masks-and-imagedatagenerator-class
def dice_coeff(y_true, y_pred):
    #print("y_true array shape: " + str(y_true.shape))
    #print("y_pred array shape: " + str(y_pred.shape))

    smooth = 1.
    y_true_f = keras.flatten(y_true)
    y_pred_f = keras.flatten(y_pred)

    #print("y_true_f array shape: " + str(y_true_f.shape))
    #print("y_pred_f array shape: " + str(y_pred_f.shape))
    #input("waiting")

    intersection = keras.sum(y_true_f * y_pred_f)
    return 1 - (2. * intersection + smooth) / (keras.sum(y_true_f) + keras.sum(y_pred_f) + smooth)


#def iou_coeff(target, prediction):
#  #https://www.jeremyjordan.me/evaluating-image-segmentation-models/
#  intersection = np.logical_and(target, prediction)
#  union = np.logical_or(target, prediction)
#  return (np.sum(intersection) / np.sum(union))

# https://stackoverflow.com/questions/53248099/keras-image-segmentation-using-grayscale-masks-and-imagedatagenerator-class
def iou_coeff(y_true, y_pred, smooth=1):
  intersection = keras.sum(keras.abs(y_true * y_pred), axis=[1,2,3])
  union = keras.sum(y_true,[1,2,3])+keras.sum(y_pred,[1,2,3])-intersection
  iou = keras.mean((intersection + smooth) / (union + smooth), axis=0)
  return iou


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

            x_set = x.reshape((-1, 320, 480, 3))
            #y_shrunk = np.zeros((320,480), dtype=np.uint8)
            y_shrunk = np.zeros((320,480,1), dtype=np.uint8)
            #y_shrunk = np.zeros((320,480,2), dtype=np.uint8)
            #y_shrunk = np.zeros((320,480,3), dtype=np.uint8)

            for row_small in range(0, 320):
                for col_small in range(0, 480):
                    row = row_small*3
                    col = col_small*3
                    total_value = y[row,col] + y[row,col+1] + y[row,col+2] + y[row+1,col] + y[row+1,col+1] + y[row+1,col+2] + y[row+2,col] + y[row+2,col+1] + y[row+2,col+2]
                    #y_shrunk[row_small,col_small] = round(total_value / 9)
                    y_shrunk[row_small,col_small,0] = round(total_value / 9)
                    #avg_val = round(total_value / 9)
                    #y_shrunk[row_small,col_small,0] = avg_val
                    #y_shrunk[row_small,col_small,1] = avg_val
                    #y_shrunk[row_small,col_small,2] = avg_val

            #print("shrunk array shape: " + str(y_shrunk.shape))

            y_set = np.array(y, dtype=np.uint8)
            #y_set = y_shrunk.reshape((-1, 320, 480))
            y_set = y_shrunk.reshape((-1, 320, 480, 1))
            #y_set = y_shrunk.reshape((-1, 320, 480, 2))
            #y_set = y_shrunk.reshape((-1, 320, 480, 3))

            #print("x_set array shape: " + str(x_set.shape))
            #print("y_set array shape: " + str(y_set.shape))
            #print(x_set.shape) print(y_set.shape) print(y_set) print("--") print(str(counter)) counter += 1

            yield (x_set, y_set)
        

# UNet:
# https://github.com/zhixuhao/unet

#num_fl = 9
#num_fl_val = 9
num_fl = 130
num_fl_val = 32

in_path_unseg = '/worksite/inprime'
in_path_unseg_val = '/worksite/inprimeval'
in_path_seg = '/worksite/inseg'
in_path_seg_val = '/worksite/insegval'
out_path = '/outgoing/'
#batchsize = 3
batchsize = 1
#input_size = (960, 1440, 3)
input_size = (320, 480, 3)
neuron_default = 64

input_layer = Input(input_size)

conv1 = Conv2D(neuron_default, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(input_layer)
conv1 = Conv2D(neuron_default, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

conv2 = Conv2D(neuron_default * 2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
conv2 = Conv2D(neuron_default * 2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

conv3 = Conv2D(neuron_default * 4, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
conv3 = Conv2D(neuron_default * 4, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

conv4 = Conv2D(neuron_default * 8, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
conv4 = Conv2D(neuron_default * 8, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
drop4 = Dropout(0.5)(conv4)
pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

conv5 = Conv2D(neuron_default * 16, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
conv5 = Conv2D(neuron_default * 16, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
#drop5 = Dropout(0.5)(conv5)

#up6 = Conv2D(neuron_default * 8, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop5))
#up6 = Conv2D(neuron_default * 8, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv5))
up6 = Conv2DTranspose(neuron_default * 8, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv5))
merge6 = concatenate([drop4,up6], axis = 3)
conv6 = Conv2D(neuron_default * 8, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
conv6 = Conv2D(neuron_default * 8, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)

#up7 = Conv2D(neuron_default * 4, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6))
up7 = Conv2DTranspose(neuron_default * 4, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6))
merge7 = concatenate([conv3,up7], axis = 3)
conv7 = Conv2D(neuron_default * 4, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
conv7 = Conv2D(neuron_default * 4, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)

#up8 = Conv2D(neuron_default * 2, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7))
up8 = Conv2DTranspose(neuron_default * 2, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7))
merge8 = concatenate([conv2,up8], axis = 3)
conv8 = Conv2D(neuron_default * 2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
conv8 = Conv2D(neuron_default * 2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)

#up9 = Conv2D(neuron_default, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8))
up9 = Conv2DTranspose(neuron_default, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8))
merge9 = concatenate([conv1,up9], axis = 3)
conv9 = Conv2D(neuron_default, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
conv9 = Conv2D(neuron_default, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
#conv9 = Conv2D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
conv9 = Conv2D(3, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)

# Input shape: 4D tensor with shape: (samples, channels, rows, cols)
# Output shape: 4D tensor with shape: (samples, filters, new_rows, new_cols) 
#   if data_format='channels_first' or 4D tensor with shape: (samples, new_rows, new_cols, filters) 
#   if data_format='channels_last'. rows and cols values might have changed due to padding.
#conv10 = Conv2D(2, 1, activation = 'sigmoid')(conv9)
conv10 = Conv2D(1, 1, activation = 'sigmoid')(conv9)
#conv10 = Conv2D(3, 1, activation = 'sigmoid')(conv9)

model = Model(inputs = input_layer, outputs = conv10)

#loss_type = 'binary_crossentropy'
monitor_type = 'loss'

#model.compile(optimizer = Adam(lr = 1e-4), loss = loss_type, metrics = ['accuracy'])
#model.compile(optimizer = Adam(lr = 1e-4), loss = 'cosine_similarity', metrics = ['accuracy'])

#model.compile(optimizer = Adam(lr = 1e-4), loss = iou_coeff, metrics = ['accuracy'])
model.compile(optimizer = Adam(lr = 1e-4), loss = dice_coeff, metrics = ['accuracy'])


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
CHKPT = ModelCheckpoint(out_path + 'best_model_incremental.h5', 
                     monitor=monitor_type, 
                     mode='min', 
                     verbose=1, 
                     save_best_only=True)


# Set up the batch generator
batch_gen = batchmaker('/worksite/inprime/all_data/', 
                   '/worksite/inseg/all_data/', 
                   batchsize, input_size)

batch_gen_val = batchmaker('/worksite/inprimeval/all_data/', 
                   '/worksite/insegval/all_data/', 
                   batchsize, input_size)

history = model.fit_generator(batch_gen,
                    steps_per_epoch=num_fl // batchsize,
                    shuffle=True, 
                    epochs=500,
                    validation_data=batch_gen_val,
                    validation_steps=num_fl_val // batchsize,
                    callbacks=[EARLYSTOP, CHKPT])

#model.save_weights(out_path + 'last_weights.h5') 

model.save(out_path + 'last_weights.h5') 

print("Done")
