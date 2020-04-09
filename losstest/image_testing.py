#from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
#from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
#from tensorflow.keras.models import *
#from tensorflow.keras.layers import *
#from tensorflow.keras.optimizers import *
##from tensorflow.keras import backend as keras
#from tensorflow.keras import backend as K
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
loss_divisor = float(input_size[0] * input_size[1] * input_size[2])



img = 'test.tif'
test_fundus = load_img(img, target_size=input_size)
ary = image.img_to_array(test_fundus)

ary.tofile('img_ary.csv', sep=',')

f = open('ary.txt', 'w')

print(len(ary))
print(len(ary[0]))
print(len(ary[0,0]))

for x in range(len(ary)):
   for y in range(len(ary[0])):
      combined_color = (262144 * ary[x,y,0]) + (512 * ary[x,y,1]) + ary[x,y,2]
      f.write(str(combined_color) + ',')

f.close()
