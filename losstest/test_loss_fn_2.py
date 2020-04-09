from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from tensorflow.keras import backend as K
#from tensorflow.keras import backend as keras
import os
import numpy as np
import tensorflow as tf
from PIL import Image

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

# Will be used to guarantee there are no divisions by zero
epsilons = [1e-6] * batchsize
min_allowed_values = tf.convert_to_tensor(epsilons, dtype=tf.float64)

#smoothing_factor = float(input_size[0] * input_size[1])
#one_list = [1.0] * (input_size[0] * input_size[1] * input_size[2])
#ones_array = np.asarray(one_list, dtype=np.float32)

#@tf.function
#def fix_div_zero(tensor_b):
#    # Used in the custom_loss() function to remove division by zero in
#    # the alpha and beta factors
#
#    # We will put together a corrected set of answers here
#    buildloc = [1e-5] * batchsize
#    #buildloc = np.array([1] * batchsize)
#    count = 0
#
#    for e in tensor_b:
#        if e > 0:
#            buildloc[count] = e
#
#    return tf.convert_to_tensor(buildloc, dtype=tf.float64)
#
#    #return tf.ones_like(x, dtype=tf.float64)
#
#@tf.function
#def func_false():
#    # Used in the custom_loss() function to remove division by zero
#    return x

def custom_loss(y_true, y_pred):
    # y_true: ground truth.  y_pred: predictions
    #
    # The logic behind this is that there are two factors in correctness:
    # 1) How correct is the prediction?  This is calculated by:
    #       A = prediction * ground truth.
    #    This gives a "mask" which zeroes out all the incorrect pixels in
    #    the prediction, leaving only the values [0..1] that it predicted
    #    for pixels that are in the ground truth mask.
    #    We then take |A|, the sum of all the predictions [0..1] within the
    #    corrrect mask.
    #       |GT| = Total number of pixels in the correct (ground truth) mask
    #    Factor alpha, then, is |A| / |GT|.
    #    Factor alpha is 0 when the prediction is awful, and 1 when it is perfect.
    #
    # 2) How incorrect is the prediction?  This is calculated by:
    #       |B| = |prediction| - |A|
    #    B would be the mask of all pixels that aren't in the ground truth.
    #    We don't need to calculate the mask; we just calculate the sum of
    #    all incorrect predictions.
    #    Factor beta, then, is |B| / |predictions|.
    #    Factor beta is 0 when the prediction is perfect, and 1 when it is awful.
    #
    # Now we can calculate the loss:
    #    loss = [abs(alpha - 1) + beta] / 2
    #
    # We will calculate the loss for each class separately and then take the mean of them
    # so that we don't have the largest class dominate the loss calculations.  (The smaller
    # ones will have much more weight than usual, which is expected from our data, where
    # the cup and disc are tiny compared to the background.)

    y_true_real = tf.dtypes.cast(y_true, tf.float64)
    y_pred_real = tf.dtypes.cast(y_pred, tf.float64)

    #print(y_true_real)

    # Separate the ground truth and prediction into their classes.
    y_true_cls0 = y_true_real[:,:,:,0]
    y_true_cls1 = y_true_real[:,:,:,1]
    y_true_cls2 = y_true_real[:,:,:,2]

    y_pred_cls0 = y_pred_real[:,:,:,0]
    y_pred_cls1 = y_pred_real[:,:,:,1]
    y_pred_cls2 = y_pred_real[:,:,:,2]

    # y_true is 1 for correct pixels, and 0 for incorrect ones, so positive_diff is
    # a "mask" of the right values in y_pred.  And negative_diff is a "mask" of the wrong
    # values in y_pred.  The size variables contain the sums of those masks.

    # Calculate A, GT, and factor alpha.
    GT_size_0 = tf.reduce_sum(y_true_cls0, axis=(1,2))
    A_size_0 = tf.reduce_sum((y_true_cls0 * y_pred_cls0), axis=(1,2))
    #alpha_0 = A_size_0 / GT_size_0
    alpha_0 = (1 + A_size_0) / (1 + GT_size_0)

    GT_size_1 = tf.reduce_sum(y_true_cls1, axis=(1,2))
    A_size_1 = tf.reduce_sum((y_true_cls1 * y_pred_cls1), axis=(1,2))
    #alpha_1 = A_size_1 / GT_size_1
    alpha_1 = (1 + A_size_1) / (1 + GT_size_1)

    GT_size_2 = tf.reduce_sum(y_true_cls2, axis=(1,2))
    A_size_2 = tf.reduce_sum((y_true_cls2 * y_pred_cls2), axis=(1,2))
    #alpha_2 = A_size_2 / GT_size_2
    alpha_2 = (1 + A_size_2) / (1 + GT_size_2)

    """
    # If one of the classes is all zeroes,
    #   it is very bad - discourage it.
    zero_size = tf.zeros_like(GT_size_0)                     # for comparison
    zero_alpha = tf.zeros_like(alpha_0, dtype=tf.float64)    # The "you are terrible" zero
    if tf.math.equal(GT_size_0, zero_size):
        alpha_0 = zero_alpha
    if tf.math.equal(GT_size_1, zero_size):
        alpha_1 = zero_alpha
    if tf.math.equal(GT_size_2, zero_size):
        alpha_2 = zero_alpha
    """


    print("-----------")
    print("GT")
    sess = tf.InteractiveSession()
    print(GT_size_0.eval())
    sess.close()
    print("A")
    sess = tf.InteractiveSession()
    print(A_size_0)
    print(A_size_0.eval())
    sess.close()


    #positive_diff = y_true_cls0 * y_pred_cls0
    #B_size_0 = A_size_0 - tf.reduce_sum(positive_diff, axis=(1,2))

    # Calculate B, prediction, and factor beta.
    pred_size_0 = tf.reduce_sum(y_pred_cls0, axis=(1,2))
    B_size_0 = pred_size_0 - A_size_0
    beta_0 = (1 + B_size_0) / (1 + pred_size_0)
    #beta_0 = B_size_0 / pred_size_0

    pred_size_1 = tf.reduce_sum(y_pred_cls1, axis=(1,2))
    B_size_1 = pred_size_1 - A_size_1
    beta_1 = (1 + B_size_1) / (1 + pred_size_1)
    #beta_1 = B_size_1 / pred_size_1

    pred_size_2 = tf.reduce_sum(y_pred_cls2, axis=(1,2))
    B_size_2 = pred_size_2 - A_size_2
    beta_2 = (1 + B_size_2) / (1 + pred_size_2)

    """
    # If one of the classes is all zeroes,
    #   it is very bad - discourage it.
    zero_size = tf.zeros_like(pred_size_0)               # for comparison
    ones_beta = tf.ones_like(beta_0, dtype=tf.float64)   # The "you are terrible" ones
    if tf.math.equal(beta_size_0, zero_size):
        beta_0 = ones_beta
    if tf.math.equal(beta_size_1, zero_size):
        beta_1 = ones_beta
    if tf.math.equal(beta_size_2, zero_size):
        beta_2 = ones_beta
    """

    print("pred before cond")
    print(pred_size_0)
    sess = tf.InteractiveSession()
    print(pred_size_0.eval())
    sess.close()


    # If one of the classes is all zeroes,
    #   it is very bad - discourage it.

    #epsilons = [1e-6] * batchsize
    #min_allowed_values = tf.convert_to_tensor(epsilons, dtype=tf.float64)

    # WORKING ON THIS -----------------------
    #pred_size_0 = tf.math.maximum(pred_size_0, min_allowed_values)
    #pred_size_1 = tf.math.maximum(pred_size_1, min_allowed_values)
    #pred_size_2 = tf.math.maximum(pred_size_2, min_allowed_values)


    #pred_size_0 = fix_div_zero(pred_size_0)
    #pred_size_1 = fix_div_zero(pred_size_1)
    #pred_size_2 = fix_div_zero(pred_size_2)

    #zero_size = tf.zeros_like(pred_size_0)        # for comparison
    #pred_size_0 = tf.cond(tf.math.equal(pred_size_0, zero_size), lambda: tf.ones_like(x, dtype=tf.float64), lambda: x)
    #pred_size_1 = tf.cond(tf.math.equal(pred_size_0, zero_size), lambda: tf.ones_like(x, dtype=tf.float64), lambda: x)
    #pred_size_2 = tf.cond(tf.math.equal(pred_size_0, zero_size), lambda: tf.ones_like(x, dtype=tf.float64), lambda: x)

    print("pred")
    print(pred_size_0)
    sess = tf.InteractiveSession()
    print(pred_size_0.eval())
    sess.close()
    print("B")
    sess = tf.InteractiveSession()
    print(B_size_0.eval())
    sess.close()
    print("alpha")
    sess = tf.InteractiveSession()
    print(alpha_0.eval())
    sess.close()
    print("beta")
    sess = tf.InteractiveSession()
    print(beta_0.eval())
    sess.close()
    print("-----------")

    # Calculate final losses
    loss_0 = (K.abs(1 - alpha_0) + beta_0) / 2.0
    loss_1 = (K.abs(1 - alpha_1) + beta_1) / 2.0
    loss_2 = (K.abs(1 - alpha_2) + beta_2) / 2.0

    """
    # WORKING ON THIS TOO -----------------------
    # If one of the factors (alpha or beta)  is nan,
    # that means we had a class with zero pixels marked
    # as that class.  This is bad; fix it so that this
    # class is given the worst possible value.  We 
    # could fix this by adding a factor (e.g. 1) to
    # the numerator and denominator of the division, but
    # this would 
    import tensorflow as tf
    with tf.Session():
        has_nans = tf.constant([float('NaN'), 1.])
        print(tf.where(tf.is_nan(has_nans), tf.zeros_like(has_nans), has_nans).eval()
    """

    sess = tf.InteractiveSession()
    print(tf.reshape(tf.reduce_mean(loss_0 + loss_1 + loss_2), (-1,1,1,1)).eval())
    sess.close()

    #loss = (loss_0 + loss_1 + loss_2) / 3.0
    #return tf.reshape(tf.reduce_sum(loss_0 + loss_1 + loss_2) / 3.0, (-1,1,1,1))
    return tf.reshape(tf.reduce_mean(loss_0 + loss_1 + loss_2) / 3.0, (-1,1,1,1))

    """
    print("numerator, then denominator")
    sess = tf.InteractiveSession()
    print(tf.reduce_sum((y_true_cls0 * y_pred_cls0), axis=(1,2)).eval())
    sess.close()
    sess = tf.InteractiveSession()
    print(tf.reduce_sum(y_true_cls0, axis=(1,2)).eval())
    sess.close()


    y_true_cls0_diff = tf.reduce_sum(y_true_cls0 * y_pred_cls0, axis=(1,2)) / tf.reduce_sum(y_true_cls0, axis=(1,2))
    y_true_cls1_diff = tf.reduce_sum(y_true_cls1 * y_pred_cls1, axis=(1,2)) / tf.reduce_sum(y_true_cls1, axis=(1,2))
    y_true_cls2_diff = tf.reduce_sum(y_true_cls2 * y_pred_cls2, axis=(1,2)) / tf.reduce_sum(y_true_cls2, axis=(1,2))

    #print("----")
    #print("y_true_cls0_diff")
    #print(y_true_cls0_diff)
    #print("----")

    ret = tf.reshape(tf.reduce_sum(y_true_cls0_diff + y_true_cls1_diff + y_true_cls2_diff) / 3.0, (-1,1,1,1))
    #print("ret")
    #print(ret)
    print("&&&&&&")

    return ret
    """

def soft_loss(y_true, y_pred):
    return 1 - custom_loss(y_true, y_pred)


print("Start")

# Make a test "batch"
y_test = np.zeros((1,320,480,3), dtype=np.uint8)
x_test = np.zeros((1,320,480,3), dtype=np.uint8)

for x in range(130, 145):
    for y in range(130, 145):
        x_test[0,x,y,0] = 1
        y_test[0,x,y,0] = 1
for x in range(170, 208):
    for y in range(120, 152):
        x_test[0,x,y,1] = 1
        y_test[0,x,y,1] = 1
for x in range(110, 137):
    for y in range(130, 145):
        x_test[0,x,y,2] = 1
        y_test[0,x,y,2] = 1



#tf.convert_to_tensor(x_test, np.float64)
#tf.convert_to_tensor(y_test, np.float64)

#reply = custom_loss(x_test, y_test)
reply = custom_loss(tf.convert_to_tensor(x_test, np.float64), tf.convert_to_tensor(y_test, np.float64))
reply_numpy = K.eval(reply)
print('Same: ' +  str(reply_numpy))

"""
for xmod in range(4):
    for ymod in range(4):
        print("-----")
        y_test = np.zeros((1,320,480,3), dtype=np.uint8)
        x_test = np.zeros((1,320,480,3), dtype=np.uint8)

        # Add detail to x_test
        for x in range(100, 145+xmod):
            for y in range(120, 170+ymod):
                x_test[0,x,y,1] = 1

        for x in range(130, 140+xmod):
            for y in range(130, 140+ymod):
                x_test[0,x,y,2] = 1

        # Add detail to y_test
        for x in range(100, 150):
            for y in range(120, 175):
                y_test[0,x,y,1] = 1

        for x in range(130, 145):
            for y in range(130, 145):
                y_test[0,x,y,2] = 1


        ##print(str(x) +  ',' + str(y))
        #x_test[0,y,x,classnum] = 1
        reply = custom_loss(tf.convert_to_tensor(x_test, np.float64), tf.convert_to_tensor(y_test, np.float64))
        #reply = custom_loss(x_test, y_test)
        reply_numpy = K.eval(reply)
        print(str(reply_numpy))
"""

print("&&&&&&&&&&&&&&&&&&&&&&&&&&&")

# What is the coeff when the prediction is half wrong?
y_test = np.zeros((1,320,480,3), dtype=np.uint8)
x_test = np.zeros((1,320,480,3), dtype=np.uint8)

for classnum in range(3):
    for x in range(230, 250):
        for y in range(150, 170):
            x_test[0,y,x,classnum] = 1

#for x in range(170, 208):
#    for y in range(120, 152):
#        x_test[0,x,y,1] = 1
#for x in range(110, 137):
#    for y in range(130, 145):
#        x_test[0,x,y,2] = 1


for classnum in range(3):
    for x in range(240):
        for y in range(160):
            y_test[0,y,x,classnum] = 1


#reply = custom_loss(x_test, y_test)
reply = custom_loss(tf.convert_to_tensor(x_test, np.float64), tf.convert_to_tensor(y_test, np.float64))
reply_numpy = K.eval(reply)
print('Half: ' +  str(reply_numpy))

# What is the coeff when the prediction is "perfectly wrong"?
x_test = np.zeros((1,320,480,3), dtype=np.uint8)
#y_test = np.zeros((1,320,480,3), dtype=np.uint8)
y_test = np.ones_like(x_test, dtype=np.uint8)

for classnum in range(3):
    for x in range(240):
        for y in range(160):
            x_test[0,y,x,classnum] = 1

for classnum in range(3):
    for x in range(240):
        for y in range(160):
            y_test[0,y,x,classnum] = 0

#for classnum in range(3):
#    for x in range(241, 480):
#        for y in range(161,320):
#            y_test[0,y,x,classnum] = 1

#reply = custom_loss(x_test, y_test)
reply = custom_loss(tf.convert_to_tensor(x_test, np.float64), tf.convert_to_tensor(y_test, np.float64))
reply_numpy = K.eval(reply)
print('Opposite: ' +  str(reply_numpy))



print()

print()

print("Done")

