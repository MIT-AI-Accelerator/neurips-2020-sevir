import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.vgg16 import preprocess_input

# Download this file using keras first.
_default_vgg_weights = f'{os.environ["HOME"]}/.keras/models/vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5'


def get_vgg_model(input_shape=(384,384,3), layer_names=['block5_conv4'], weights):
    # get pre-trained VGG model
    try:
        vgg = tf.keras.applications.VGG19(input_shape=input_shape, include_top=False, weights=_default_vgg_weights)
    except:
        vgg = tf.keras.applications.VGG19(input_shape=input_shape, include_top=False, weights='imagenet')
    vgg.trainable = False  
    for l in vgg.layers:
        l.trainable=False
    outputs = [vgg.get_layer(name).output for name in layer_names]
    model = tf.keras.Model(inputs=vgg.input, outputs=outputs)
    return model, len(layer_names)

def get_vgg_content(input_shape=(384,384,3)):
    return get_vgg_model(input_shape, layer_names=['block5_conv4'])

def get_vgg_style(input_shape=(384,384,3)):
    return get_vgg_model(input_shape, layer_names=['block1_conv1', 'block2_conv1', 'block3_conv1', 
                                                   'block4_conv1', 'block5_conv1'])

def gram_matrix(input_tensor):
  result = tf.linalg.einsum('bijc,bijd->bcd', input_tensor, input_tensor)
  input_shape = tf.shape(input_tensor)
  num_locations = tf.cast(input_shape[1]*input_shape[2], tf.float32)
  return result/(num_locations)

content_model,_ = get_vgg_content(input_shape=(384,384,3))

def vgg_content_loss(X,Y):
    """
    Expects grayscale images on 0-255 scale (N,384,384,1)
    """
    # create pseudo-rgb
    x_rgb = tf.image.grayscale_to_rgb(tf.expand_dims(x, axis=-1))
    y_rgb = tf.image.grayscale_to_rgb(tf.expand_dims(y, axis=-1))

    # preprocess for vgg
    x_rgb = tf.keras.applications.vgg16.preprocess_input(x_rgb) # what are the data values here ? 
    y_rgb = tf.keras.applications.vgg16.preprocess_input(y_rgb) 

    # calculate style and content features
    x_feats = content_model(x_rgb)
    y_feats = content_model(y_rgb)

    # style tensors
    style = [tf.math.reduce_sum(tf.square(A-B)) for A,B in zip(x_style, y_style)]
    
    # "content" tensors
    content = [tf.math.reduce_sum(tf.square(x_feats[-1]-y_feats[-1]))]

    loss = tf.add_n(style+content)
    return loss



def vggloss(X,Y):
    """
    Expects grayscale images on 0-255 scale
    """

    # create pseudo-rgb
    x_rgb = tf.image.grayscale_to_rgb(tf.expand_dims(x, axis=-1))
    y_rgb = tf.image.grayscale_to_rgb(tf.expand_dims(y, axis=-1))

    # preprocess for vgg
    x_rgb = tf.keras.applications.vgg16.preprocess_input(x_rgb) # what are the data values here ? 
    y_rgb = tf.keras.applications.vgg16.preprocess_input(y_rgb) 

    # calculate style and content features
    x_feats = stylemodel(x_rgb)
    y_feats = stylemodel(y_rgb)
    x_style = [gram_matrix(A) for A in x_feats[:-1]]
    y_style = [gram_matrix(A) for A in y_feats[:-1]]

    # style tensors
    style = [tf.math.reduce_sum(tf.square(A-B)) for A,B in zip(x_style, y_style)]
    
    # "content" tensors
    content = [tf.math.reduce_sum(tf.square(x_feats[-1]-y_feats[-1]))]

    loss = tf.add_n(style+content)
    return loss

def vggloss_scaled(X,Y):
    # this loss function is used when combining MSE + VGG style/content
    MEAN = 33.44
    SCALE = 47.54
    # these weights should be adjusted
    NUM_LAYERS = np.float32(5) # 
    # these are based on mean : STYLE_WEIGHTS = np.array([2.6e6, 7.1e8, 4.5e9, 8e9, 8e7], dtype=np.float32)
    # these are based on variance :
    STYLE_WEIGHTS = np.array([5.550377e+10, 3.416415e+15, 2.321729e+17, 5.504242e+17, 9.636540e+13], dtype=np.float)

    # based on mean : CONTENT_WEIGHT = np.float32(1.7e6)
    # based on variance
    CONTENT_WEIGHT = np.float32(4.766082e+10)
    
    X_ustack = tf.unstack(X, num=12, axis=-1)
    Y_ustack = tf.unstack(Y, num=12, axis=-1)

    loss = 0.0
    for x,y in zip(X_ustack, Y_ustack):
        # scale back to original 0-255 range
        x = x*SCALE + MEAN
        y = y*SCALE + MEAN

        # create pseudo-rgb
        x_rgb = tf.image.grayscale_to_rgb(tf.expand_dims(x, axis=-1))
        y_rgb = tf.image.grayscale_to_rgb(tf.expand_dims(y, axis=-1))

        # preprocess for vgg
        x_rgb = tf.keras.applications.vgg16.preprocess_input(x_rgb) # what are the data values here ? 
        y_rgb = tf.keras.applications.vgg16.preprocess_input(y_rgb) 

        # calculate style and content features
        x_feats = stylemodel(x_rgb)
        y_feats = stylemodel(y_rgb)
        x_style = [gram_matrix(A) for A in x_feats[:-1]]
        y_style = [gram_matrix(A) for A in y_feats[:-1]]

        # style tensors
        style = [tf.math.divide(tf.math.reduce_sum(tf.square(A-B)), NUM_LAYERS*C)
                     for A,B,C in zip(x_style, y_style, STYLE_WEIGHTS)]
        
        # "content" tensors
        content = tf.math.divide(tf.math.reduce_sum(tf.square(x_feats[-1]-y_feats[-1])), CONTENT_WEIGHT)

        loss = loss + tf.add_n(style + [content])
    return loss




