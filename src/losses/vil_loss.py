import numpy as np
import tensorflow as tf

def getVIL(x,a,b):
    X = tf.math.multiply(tf.cast(tf.math.greater_equal(x,a), tf.float32), x)
    Y = tf.math.multiply(tf.cast(tf.math.less(x,b), tf.float32), x)
    VIL = tf.math.sign(tf.math.multiply(tf.math.floormod(X, b),Y))
    return VIL

def getOnesidedVILv2(x, a):
    X = tf.math.sign(tf.math.multiply(tf.cast(tf.math.greater_equal(x,a), tf.float32), x))
    VIL = X * 1.0
    return VIL

def getOnesidedVIL(x,a):
    X = tf.math.multiply(tf.cast(tf.math.greater_equal(x,a), tf.float32), x)
    VIL = tf.math.sign(X)
    return VIL

def vil_loss(X,Y):
    MEAN, SCALE  = 33.44, 47.54
    loss = 0.    
    # first rescale the data to original raw numbers (0-255)
    X = X*SCALE + MEAN
    Y = Y*SCALE + MEAN
    X_ustack = tf.unstack(X, num=12, axis=-1)
    Y_ustack = tf.unstack(Y, num=12, axis=-1)
    
    # for each time step predicted, calculate the DICE loss
    for x,y in zip(X_ustack, Y_ustack):        
        # calculate the DICE loss for one VIL level
        VIL_X = getOnesidedVIL(x,74.)
        VIL_Y = getOnesidedVIL(y,74.)
        loss += dice_coeff(VIL_X, VIL_Y)
    return loss

def vil_loss_multilevel(X,Y):
    MEAN, SCALE  = 33.44, 47.54
    eps = 1.
    loss = 0.
    lev = [0.0, 16.0, 31.0, 59.0, 74.0, 100.0, 133.0, 160.0, 181.0, 219.0, 255.0]
    # first rescale the data to original raw numbers (0-255)
    X = X*SCALE + MEAN
    Y = Y*SCALE + MEAN
    X_ustack = tf.unstack(X, num=12, axis=-1)
    Y_ustack = tf.unstack(Y, num=12, axis=-1)
    
    # for each time step predicted, calculate the loss
    for x,y in zip(X_ustack, Y_ustack):
        # calculate the DICE loss for each VIL band
        for a,b in zip(lev[:-1], lev[1:]):
            VIL_X = getVIL(x,a,b)
            VIL_Y = getVIL(y,a,b)
            loss += dice_coeff(VIL_X, VIL_Y)
    return loss


# from https://stackoverflow.com/questions/51973856/how-is-the-smooth-dice-loss-differentiable
# and https://github.com/jocicmarko/ultrasound-nerve-segmentation/blob/master/train.py
def dice_coeff(y_true, y_pred):
    smooth = 1.
    # Flatten
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)    
    score = (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)
    return score

def dice_loss(y_true, y_pred):
    loss = 1 - dice_coeff(y_true, y_pred)
    return loss

