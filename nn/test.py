#from tensorflow.keras.models import Sequential
#from tensorflow.keras.layers import Conv2D, MaxPooling2D, Conv2DTranspose
#from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
#from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from tensorflow.keras import backend as keras
import os
import numpy as np
import tensorflow as tf

# UNet:
# https://github.com/zhixuhao/unet


num_fl = 130
num_fl_val = 32
in_path_unseg = '/worksite/inprime/'
in_path_unseg_val = '/worksite/inprimeval/'
in_path_seg = '/worksite/inseg/'
in_path_seg_val = '/worksite/insegval/'
out_path = '/outgoing/'
#batchsize = 3
batchsize = 1
#input_size = (960, 1440, 3)
input_size = (320, 480, 3)


# A Python generator that will give the fit_generator training data in batches.
# Details at: https://wiki.python.org/moin/Generators
def batchmaker_train():

    x_files = sorted(os.listdir(in_path_unseg))
    y_files = sorted(os.listdir(in_path_seg))
    
    if batchsize > num_fl:
        print("Oh...COME ON!  batchsize too large.")
        # And then crash.  But this won't happen.  Right?
    
    # The infinite loop is part of how generators work.  The fit_generator needs to
    # always have data available, so we loop forever.
    while True:
        # (Re)start at the head of the files.
        batch_head = 0                              # Start of current batch in the files
        batch_end = batch_head + batchsize          # End of the batch
        
        while batch_end < num_fl:
            x_flname = x_files[batch_head:batch_end]
            y_flname = y_files[batch_head:batch_end]

            x_set = []
            y_set = []
            for flname in x_files:
                if not os.path.isdir(os.path.join(in_path_unseg, flname)):
                    x_set.append(img_to_array(tf.image.resize(load_img(in_path_unseg + flname), [320,480])))
                    #x_set.append(img_to_array(load_img(in_path_unseg + flname)))
            for flname in y_files:
                if not os.path.isdir(os.path.join(in_path_seg, flname)):
                    y_set.append(img_to_array(tf.image.resize(load_img(in_path_seg + flname), [320,480])))
                    #y_set.append(img_to_array(load_img(in_path_seg + flname)))
                    #y_set.append(load_img(in_path_seg + flname))
            
            batch_head = batch_end
            batch_end = batch_head + batchsize 
            
            x_set = np.array(x_set)
            y_set = np.array(y_set)
            
            yield (x_set, y_set)

# A Python generator that will give the fit_generator test data in batches.
# Details at: https://wiki.python.org/moin/Generators
def batchmaker_test():

    x_files = sorted(os.listdir(in_path_unseg_val))
    y_files = sorted(os.listdir(in_path_seg_val))
    
    if batchsize > num_fl_val:
        print("Oh...COME ON!  batchsize too large.")
        # And then crash.  But this won't happen.  Right?
    
    # The infinite loop is part of how generators work.  The fit_generator needs to
    # always have data available, so we loop forever.
    while True:
        # (Re)start at the head of the files.
        batch_head = 0                              # Start of current batch in the files
        batch_end = batch_head + batchsize          # End of the batch
        
        while batch_end < num_fl_val:
            x_flname = x_files[batch_head:batch_end]
            y_flname = y_files[batch_head:batch_end]

            x_set = []
            y_set = []
            for flname in x_files:
                if not os.path.isdir(os.path.join(in_path_unseg_val, flname)):
                    x_set.append(tf.image.resize(load_img(in_path_unseg_val + flname), [320,480]))
                    #x_set.append(load_img(in_path_unseg_val + flname))
            for flname in y_files:
                if not os.path.isdir(os.path.join(in_path_seg_val, flname)):
                    y_set.append(tf.image.resize(load_img(in_path_seg_val + flname), [320,480]))
                    #y_set.append(load_img(in_path_seg_val + flname))
            
            batch_head = batch_end
            batch_end = batch_head + batchsize 
            
            x_set = np.array(x_set)
            y_set = np.array(y_set)
            
            yield (x_set, y_set)

#def unet(pretrained_weights = None,input_size = (256,256,1)):

input_layer = Input(input_size)

conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(input_layer)
conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

#conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
#conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
#pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
#
#conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
#conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
#drop4 = Dropout(0.5)(conv4)
#pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)
#
#conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
#conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
#drop5 = Dropout(0.5)(conv5)
#
#up6 = Conv2D(512, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop5))
#merge6 = concatenate([drop4,up6], axis = 3)
#conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
#conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)
#
#up7 = Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6))
#merge7 = concatenate([conv3,up7], axis = 3)
#conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
#conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)

#up8 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv2))
#merge8 = concatenate([conv2,up8], axis = 3)
#conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
#conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)

up9 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv2))
merge9 = concatenate([conv1,up9], axis = 3)
conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
conv9 = Conv2D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)

conv10 = Conv2D(1, 1, activation = 'sigmoid')(conv9)

model = Model(inputs = input_layer, outputs = conv10)

#model.compile(optimizer = Adam(lr = 1e-4), loss = 'binary_crossentropy', metrics = ['accuracy'])

model.compile(optimizer = Adam(lr = 1e-4), loss = 'cosine_similarity', metrics = ['accuracy'])

print("++++++++++++++")
print(model.count_params())
print("++++++++++++++")
print(model.summary())
print("++++++++++++++")
os.system('free -m')
print("++++++++++++++")
os.system('vmstat -s')
print("++++++++++++++")


"""
model.compile(loss='sparse_categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
"""

"""
EARLYSTOP = EarlyStopping(patience=50, 
                          monitor='val_categorical_accuracy', 
                          restore_best_weights=True)
EARLYSTOP = EarlyStopping(patience=50, 
                          monitor='binary_crossentropy', 
                          restore_best_weights=True)
"""

EARLYSTOP = EarlyStopping(patience=50, 
                          monitor='cosine_similarity', 
                          restore_best_weights=True)

"""
CHKPT = ModelCheckpoint(out_path + 'best_model_incremental.h5', 
                     monitor='val_categorical_accuracy', 
                     mode='max', 
                     verbose=1, 
                     save_best_only=True)
CHKPT = ModelCheckpoint(out_path + 'best_model_incremental.h5', 
                     monitor='binary_crossentropy', 
                     mode='max', 
                     verbose=1, 
                     save_best_only=True)
"""

# Save off the very best model we can find; avoids overfitting.
CHKPT = ModelCheckpoint(out_path + 'best_model_incremental.h5', 
                     monitor='cosine_similarity', 
                     mode='max', 
                     verbose=1, 
                     save_best_only=True)

"""
history = model.fit_generator(batchmaker_train(),
                    steps_per_epoch=num_fl // batchsize,
                    shuffle=True, 
                    epochs=500,
                    validation_data=batchmaker_test(),
                    validation_steps=num_fl_val // batchsize,
                    callbacks=[EARLYSTOP, CHKPT])
"""

history = model.fit_generator(batchmaker_train(),
                    steps_per_epoch=num_fl // batchsize,
                    shuffle=True, 
                    epochs=500,
                    validation_data=batchmaker_test(),
                    validation_steps=num_fl_val // batchsize)

model.save_weights(out_path + 'last_weights.h5') 
