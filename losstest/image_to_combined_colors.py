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

from os import listdir
from os.path import isfile, join


num_fl = 130
num_fl_val = 32

path_fundus = '/local/repository/inprime/'
path_fundus_val = '/local/repository/inprimeval/'
path_mask = '/local/repository/inseg/'
path_mask_val = '/local/repository/insegval/'
#path_fundus = '/worksite/fundus/'
#path_fundus_val = '/worksite/fundusvalidate/'
#path_mask = '/worksite/mask/'
#path_mask_val = '/worksite/maskvalidate/'

path_fundus_out = '/worksite/fundus/'
path_fundus_val_out = '/worksite/fundusvalidate/'
path_mask_out = '/worksite/mask/'
path_mask_val_out = '/worksite/maskvalidate/'
#path_fundus_out = path_fundus + 'out/'
#path_fundus_val_out = path_fundus_val + 'out/'
#path_mask_out = path_mask + 'out/'
#path_mask_val_out = path_mask_val + 'out/'

out_path = '/outgoing/'

input_size = (320, 480, 3)
output_size = (320, 480)


# A Python generator that will give the fit_generator data in batches.
def main():
    ## Make directories if they don't exist
    #try:
    #    os.mkdir(path_fundus_out)
    #    os.mkdir(path_fundus_val_out)
    #    os.mkdir(path_mask_out)
    #    os.mkdir(path_mask_val_out)
    #except OSError:
    #    print("Directories already exist")

    # Make a list of the file names in the main eye image directory.
    files = [f for f in listdir(path_fundus) if isfile(join(path_fundus, f))]
    #files = os.listdir(path_fundus)
    file_size = len(files) - 1

    array_combined = np.ndarray(output_size, dtype=np.double)

    for file_nm in files:
        print(file_nm)
        img = path_fundus + file_nm
        test_fundus = load_img(img, target_size=input_size)
        ary = image.img_to_array(test_fundus)

        # Make the three color layers into one layer.
        for x in range(len(ary)):
            for y in range(len(ary[0])):
               combined_color = float((262144 * ary[x,y,0]) + (512 * ary[x,y,1]) + ary[x,y,2])
               array_combined[x,y] = combined_color

        np.save(path_fundus_out + file_nm, array_combined)

        # Get the target data, which is a saved numpy ndarray
        aryfile = path_mask + file_nm
        y = np.load(aryfile)

        y_shrunk = np.zeros(output_size, dtype=np.double)

        for row_small in range(0, input_size[0]):
            for col_small in range(0, input_size[1]):
                row = row_small * 3
                col = col_small * 3
                total_value = y[row,col] + y[row,col+1] + y[row,col+2] + y[row+1,col] + y[row+1,col+1] + y[row+1,col+2] + y[row+2,col] + y[row+2,col+1] + y[row+2,col+2]

                avg_val = round(total_value / 9.0)
                if avg_val == 0:
                    y_shrunk[row_small,col_small] = 1.0
                elif avg_val == 1:
                    y_shrunk[row_small,col_small] = 2.0
                else:
                    y_shrunk[row_small,col_small] = 3.0

        np.save(path_mask_out + file_nm, y_shrunk)

    ###################################################
    # Now take care of the files in the validation set.

    # Make a list of the file names in the main eye image directory.
    files = [f for f in listdir(path_fundus_val) if isfile(join(path_fundus_val, f))]
    #files = os.listdir(path_fundus_val)
    file_size = len(files) - 1

    array_combined = np.ndarray(output_size, dtype=np.double)
    #array_combined = np.ndarray(input_size, dtype=np.double)

    for file_nm in files:
        print(file_nm)
        img = path_fundus_val + file_nm
        test_fundus = load_img(img, target_size=input_size)
        ary = image.img_to_array(test_fundus)

        for x in range(len(ary)):
            for y in range(len(ary[0])):
               combined_color = float((262144 * ary[x,y,0]) + (512 * ary[x,y,1]) + ary[x,y,2])
               array_combined[x,y] = combined_color

        np.save(path_fundus_val_out + file_nm, array_combined)

        # Get the target data, which is a saved numpy ndarray
        aryfile = path_mask_val + file_nm
        y = np.load(aryfile)

        y_shrunk = np.zeros(output_size, dtype=np.double)
        #y_shrunk = np.zeros((input_size[0], input_size[1]), dtype=np.double)

        for row_small in range(0, input_size[0]):
            for col_small in range(0, input_size[1]):
                row = row_small * 3
                col = col_small * 3
                total_value = y[row,col] + y[row,col+1] + y[row,col+2] + y[row+1,col] + y[row+1,col+1] + y[row+1,col+2] + y[row+2,col] + y[row+2,col+1] + y[row+2,col+2]

                avg_val = round(total_value / 9.0)
                if avg_val == 0:
                    y_shrunk[row_small,col_small] = 1.0
                elif avg_val == 1:
                    y_shrunk[row_small,col_small] = 2.0
                else:
                    y_shrunk[row_small,col_small] = 3.0

        np.save(path_mask_val_out + file_nm, y_shrunk)

    print("Done")

# Launch!
main()
