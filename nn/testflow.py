from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
#from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from tensorflow.keras import backend as keras
import os
import numpy as np
import tensorflow as tf


#https://github.com/keras-team/keras/issues/5720
def combine_generator(gen1, gen2):
    while True:
        yield(gen1.next(), gen2.next())

# https://stackoverflow.com/questions/53248099/keras-image-segmentation-using-grayscale-masks-and-imagedatagenerator-class
def dice_coeff(y_true, y_pred):
    smooth = 1.
    y_true_f = keras.flatten(y_true)
    y_pred_f = keras.flatten(y_pred)
    intersection = keras.sum(y_true_f * y_pred_f)
    return 1 - (2. * intersection + smooth) / (keras.sum(y_true_f) + keras.sum(y_pred_f) + smooth)

# https://stackoverflow.com/questions/53248099/keras-image-segmentation-using-grayscale-masks-and-imagedatagenerator-class
def iou_coeff(y_true, y_pred, smooth=1):
  intersection = keras.sum(keras.abs(y_true * y_pred), axis=[1,2,3])
  union = keras.sum(y_true,[1,2,3])+keras.sum(y_pred,[1,2,3])-intersection
  iou = keras.mean((intersection + smooth) / (union + smooth), axis=0)
  return iou
        
# UNet:
# https://github.com/zhixuhao/unet

num_fl = 9
num_fl_val = 9
#num_fl = 130
#num_fl_val = 32
in_path_unseg = '/worksite/inprime'
in_path_unseg_val = '/worksite/inprimeval'
in_path_seg = '/worksite/inseg'
in_path_seg_val = '/worksite/insegval'
out_path = '/outgoing/'
#batchsize = 3
batchsize = 1
#input_size = (960, 1440, 3)
input_size = (320, 480, 3)

input_layer = Input(input_size)

conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(input_layer)
conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
drop4 = Dropout(0.5)(conv4)
pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
drop5 = Dropout(0.5)(conv5)

up6 = Conv2D(512, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop5))
merge6 = concatenate([drop4,up6], axis = 3)
conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)

up7 = Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6))
merge7 = concatenate([conv3,up7], axis = 3)
conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)

up8 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7))
merge8 = concatenate([conv2,up8], axis = 3)
conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)

up9 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8))
merge9 = concatenate([conv1,up9], axis = 3)
conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
#conv9 = Conv2D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
conv9 = Conv2D(3, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)

#conv10 = Conv2D(1, 1, activation = 'sigmoid')(conv9)
conv10 = Conv2D(3, 1, activation = 'sigmoid')(conv9)

model = Model(inputs = input_layer, outputs = conv10)

#loss_type = 'binary_crossentropy'
monitor_type = 'loss'

#model.compile(optimizer = Adam(lr = 1e-4), loss = loss_type, metrics = ['accuracy'])
model.compile(optimizer = Adam(lr = 1e-4), loss = iou_coeff, metrics = ['accuracy'])

#model.compile(optimizer = Adam(lr = 1e-4), loss = 'cosine_similarity', metrics = ['accuracy'])

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

# https://stackoverflow.com/questions/45510403/keras-for-semantic-segmentation-flow-from-directory-error
# https://stackoverflow.com/questions/53248099/keras-image-segmentation-using-grayscale-masks-and-imagedatagenerator-class
image_datagen = ImageDataGenerator(featurewise_center = True)
mask_datagen = ImageDataGenerator()
val_image_datagen = ImageDataGenerator(featurewise_center = True)
val_mask_datagen = ImageDataGenerator()

## Training data
image_generator = image_datagen.flow_from_directory(
    in_path_unseg,
    target_size=(320, 480),
    class_mode = None,
    batch_size = 1,
    seed = 123)

mask_generator = mask_datagen.flow_from_directory(
    in_path_seg,
    target_size=(320, 480),
    class_mode = None,
    batch_size = 1,
    seed = 123)

# combine generators into one which yields image and masks
#train_generator = zip(image_generator, mask_generator)
train_generator = combine_generator(image_generator, mask_generator)

## Validation data
val_image_generator = val_image_datagen.flow_from_directory(
    in_path_unseg,
    target_size=(320, 480),
    class_mode = None,
    batch_size = 1,
    seed = 123)

val_mask_generator = val_mask_datagen.flow_from_directory(
    in_path_seg,
    target_size=(320, 480),
    class_mode = None,
    batch_size = 1,
    seed = 123)

# combine generators into one which yields image and masks
#val_train_generator = zip(val_image_generator, val_mask_generator)
val_train_generator = combine_generator(val_image_generator, val_mask_generator)

#model.fit_generator(
#    train_generator,
#    steps_per_epoch = 1000,
#    epochs = 100)

#history = model.fit_generator(batchmaker_train(),
#                    steps_per_epoch=num_fl // batchsize,
#                    shuffle=True, 
#                    epochs=500,
#                    validation_data=batchmaker_test(),
#                   validation_steps=num_fl_val // batchsize)

history = model.fit_generator(train_generator,
                    steps_per_epoch=num_fl // batchsize,
                    shuffle=True, 
                    epochs=500,
                    validation_data=val_train_generator,
                    validation_steps=num_fl_val // batchsize,
                    callbacks=[EARLYSTOP, CHKPT])

model.save_weights(out_path + 'last_weights.h5') 
