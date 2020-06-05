import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.vgg16 import preprocess_input

def get_style_model(input_shape=(256,256,3)):
  # Content layer where will pull our feature maps
  content_layers = ['block5_conv2'] 

  # Style layer of interest
  style_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1', 
                  'block4_conv1', 'block5_conv1']

  layer_names = style_layers + content_layers
  num_content_layers = len(content_layers)
  num_style_layers = len(style_layers)

  # get pre-trained VGG model
  vgg = tf.keras.applications.VGG16(input_shape=input_shape, include_top=False, weights='imagenet')
  vgg.trainable = False  
  outputs = [vgg.get_layer(name).output for name in layer_names]
  model = tf.keras.Model(inputs=vgg.input, outputs=outputs)
  return model, num_style_layers, num_content_layers

stylemodel,_,_ = get_style_model(input_shape=(384,384,3))

def gram_matrix(input_tensor):
  result = tf.linalg.einsum('bijc,bijd->bcd', input_tensor, input_tensor)
  input_shape = tf.shape(input_tensor)
  num_locations = tf.cast(input_shape[1]*input_shape[2], tf.float32)
  return result/(num_locations)

# MEAN=21.11, SCALE=41.78
def vggloss(X,Y):
    MEAN = 33.44
    SCALE = 47.54

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
        style = [tf.math.reduce_sum(tf.square(A-B)) for A,B in zip(x_style, y_style)]
        
        # "content" tensors
        content = [tf.math.reduce_sum(tf.square(x_feats[-1]-y_feats[-1]))]

        loss = loss + tf.add_n(style+content)
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
