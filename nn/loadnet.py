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

# https://stackoverflow.com/questions/45510403/keras-for-semantic-segmentation-flow-from-directory-error
# https://stackoverflow.com/questions/53248099/keras-image-segmentation-using-grayscale-masks-and-imagedatagenerator-class
image_datagen = ImageDataGenerator()
#image_datagen = ImageDataGenerator(featurewise_center = True)
mask_datagen = ImageDataGenerator()
val_image_datagen = ImageDataGenerator()
#val_image_datagen = ImageDataGenerator(featurewise_center = True)
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

# Load the model from file
prevmodelfile = 'last_weights.h5'
print(' Loading model: ' + prevmodelfile)
model = load_model(prevmodelfile)
print("++++++++++++++")
print(model.count_params())
print("++++++++++++++")
print(model.summary())
print("++++++++++++++")
os.system('free -m')
print("++++++++++++++")
os.system('vmstat -s')
print("++++++++++++++")