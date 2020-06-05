import tensorflow as tf
from tensorflow.keras import layers,models
from .unet_gan import downsample

# discriminator and downsample code from :
# https://colab.research.google.com/github/tensorflow/docs/blob/master/site/en/tutorials/generative/pix2pix.ipynb
# discriminator is a patchGAN - basically a convnet with B X H x W x C output
# instead of a single true/false output
def discriminator(input_shapes, target_shapes):    
    # both input_shapes and target_shapes are expected to be a list of tuples even
    # if there's only one input and one target
    if not ( all(isinstance(A, tuple) for A in input_shapes)
                 and all(isinstance(B, tuple) for B in target_shapes) ) :
        raise ValueError('function expects inputs to be lists of tuples')
    
    initializer = tf.random_normal_initializer(0., 0.02)

    # first create all the layers corresponding to the
    # generator outptus that are inputs to the discriminator
    inputs = []
    inputs.append(layers.Input(shape=input_shapes[0], name='input_image_0'))
    for ii,shape in enumerate(input_shapes[1:]):
        inputs.append(layers.Input(shape=shape, name=f'input_image_{ii+1}'))

    # now create input layers that correspond to all the target images (this is the ground truth)
    inputs.append(layers.Input(shape=target_shapes[0], name='target_image_0'))
    for ii,shape in enumerate(target_shapes[1:]):
       inputs.append(layers.Input(shape=target_shapes[ii], name='target_image_{ii+1}'))

    # concatenate all inputs
    x = layers.concatenate(inputs)

    down1 = downsample(64, 4, False)(x) 
    down2 = downsample(128, 4)(down1) 
    down3 = downsample(256, 4)(down2) 

    zero_pad1 = layers.ZeroPadding2D()(down3)
    conv = layers.Conv2D(512, 4, strides=1, kernel_initializer=initializer,
                                  use_bias=False)(zero_pad1) 

    batchnorm1 = layers.BatchNormalization()(conv)
    leaky_relu = layers.LeakyReLU()(batchnorm1)
    zero_pad2 = layers.ZeroPadding2D()(leaky_relu) 
    output = layers.Conv2D(1, 4, strides=1, kernel_initializer=initializer, name='output')(zero_pad2) 

    model = models.Model(inputs=inputs, outputs=output)
    return model

