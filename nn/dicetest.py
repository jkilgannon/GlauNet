import tensorflow as tf
from tensorflow.keras import backend as keras
import numpy as np

def dice_coeff(y_true, y_pred):
    #print("y_true array shape: " + str(y_true.shape)) 
    print("y_pred array shape: " + str(y_pred.shape))

    smooth = 1.
    y_true_f = keras.flatten(y_true)
    y_pred_f = keras.flatten(y_pred)

    print("y_true_f array shape: " + str(y_true_f.shape)) 
    print("y_pred_f array shape: " + str(y_pred_f.shape)) 
    #input("waiting")

    intersection = keras.sum(y_true_f * y_pred_f)
    return 1 - (2. * intersection + smooth) / (keras.sum(y_true_f) + keras.sum(y_pred_f) + smooth)


fl = np.fromfile('predicted.csv', sep=",")
fl = fl.reshape((320,480,3)) 

rr = dice_coeff(fl, fl)
print(type(rr))
print(rr.shape)
#print(rr.eval() with sess.as_default())
#tf.print("tensor:", rr, output_stream=sys.stdout)
print("++++++")
with tf.Session() as sess:  print(rr.eval())
print("++++++")

print(str(dice_coeff(fl, fl)))
