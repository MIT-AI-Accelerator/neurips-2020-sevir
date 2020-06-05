import tensorflow as tf
#MEAN, SCALE  = 33.44, 47.54 # from hpec
#MEAN=21.11 # new
#SCALE=41.78 # new
NUM_TIME_STEPS = 12.0

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
    loss = 1 - dice_coeff(y_true, y_pred) #/NUM_TIME_STEPS
    return loss

def to_mask(A):
    mask = tf.clip_by_value(A, 74., 74.001)/74.
    return mask

def to_mask_ab(A, a, b):
    mask = tf.clip_by_value(A, a, b)/b
    return mask

def getOnesidedVILv2(x):
    X = tf.math.sign(tf.math.multiply(tf.cast(tf.math.greater_equal(x,74.), tf.float32), 1.0))
    VIL = X * 1.0
    return VIL

def getVIL(x,a=74.,b=255.):
    X = tf.math.multiply(tf.cast(tf.math.greater_equal(x,a), tf.float32), x)
    Y = tf.math.multiply(tf.cast(tf.math.less(x,b), tf.float32), x)
    VIL = tf.math.sign(tf.math.multiply(tf.math.floormod(X, b),Y))
    return VIL

def myloss(X,Y,MEAN=21.11,SCALE=41.78):    
    #X = tf.clip_by_value(X, 74., 255.)    
    #X = getOnesidedVILv2(X*SCALE+MEAN)
    #Y = getOnesidedVILv2(Y*SCALE+MEAN)
    X = X*SCALE + MEAN
    Y = Y*SCALE + MEAN
    A = tf.clip_by_value(X, 16., 255.)
    L1 = dice_coeff(A,Y)
    L2 = dice_coeff(A,X)
    return L2-L1

def ce(X,Y,MEAN=21.11,SCALE=41.78):
    #X = tf.clip_by_value(X, 74., 255.)/255.
    #Y = tf.clip_by_value(Y, 74., 255.)/255.
    X = (X*SCALE + MEAN)/255.
    Y = (Y*SCALE + MEAN)/255.
    return tf.keras.losses.binary_crossentropy(X,Y)

def ssim(X,Y,MEAN=21.11,SCALE=41.78):
    X = X*SCALE + MEAN
    Y = Y*SCALE + MEAN
    loss = tf.image.ssim_multiscale(X, Y, 255.)
    return loss

vgg16 = tf.keras.applications.VGG16(input_shape=(384,384,3), include_top=False, weights='imagenet')
vgg16.trainable = False  
outputs = tf.keras.layers.Flatten()(vgg16.get_layer('block5_pool').output)
feat_model = tf.keras.models.Model(inputs=vgg16.input, outputs=outputs)
def featureloss(X,Y,model=feat_model,MEAN=21.11,SCALE=41.78):
    # scale back to original 0-255 range
    X = X*SCALE + MEAN
    Y = Y*SCALE + MEAN
    X_ustack = tf.unstack(X, num=12, axis=-1)
    Y_ustack = tf.unstack(Y, num=12, axis=-1)

    loss = 0.0
    for x,y in zip(X_ustack, Y_ustack):
        # create pseudo-rgb
        x_rgb = tf.image.grayscale_to_rgb(tf.expand_dims(x, axis=-1))
        y_rgb = tf.image.grayscale_to_rgb(tf.expand_dims(y, axis=-1))

        # preprocess for vgg
        x_rgb = tf.keras.applications.vgg16.preprocess_input(x_rgb)
        y_rgb = tf.keras.applications.vgg16.preprocess_input(y_rgb) 

        # calculate VGG features
        x_feats = feat_model(x_rgb)
        y_feats = feat_model(y_rgb)

        loss += tf.keras.losses.cosine_similarity(x_feats,y_feats)
    return loss

def multiscale(X,Y,MEAN=21.11,SCALE=41.78):
    loss = 0.
    lev = [0.0, 16.0, 31.0, 59.0, 74.0, 100.0, 133.0, 160.0, 181.0, 219.0, 255.0]
    #lev = [31.0, 59.0, 74.0, 100.0, 133.0, 160.0, 181.0, 219.0]
    #lev = [74.0, 160.0, 219.0]
    X = X*SCALE + MEAN # scaling back the 0-255 range 
    Y = Y*SCALE + MEAN
    for a,b in zip(lev[:-1], lev[1:]):
    #for a in lev:
        #loss += dice_loss( to_mask_ab(X,a,b), to_mask_ab(Y,a,b) )
        loss += dice_loss( tf.clip_by_value(X,a,b)/b, tf.clip_by_value(Y,a,b)/b )
    return loss
