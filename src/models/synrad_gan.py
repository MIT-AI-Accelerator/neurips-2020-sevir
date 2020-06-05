import tensorflow as tf
from tensorflow.keras import layers,losses,models
from losses.gan_losses import generator_loss,discriminator_loss
from .unet import conv_block,encoder_block,decoder_block

def generator(norm, num_outputs=1,start_filters=32):
    ir069 = tf.keras.layers.Input(shape=(192,192,1)) 
    ir107 = tf.keras.layers.Input(shape=(192,192,1)) 
    lght  = tf.keras.layers.Input(shape=(48,48,1))
    inputs = [ir069,ir107,lght]

    # Normalize inputs
    ir069_norm = tf.keras.layers.Lambda(lambda x,mu,scale: (x-mu)/scale,
                                       arguments={'mu':norm['ir069']['shift'],'scale':norm['ir069']['scale']})(ir069)
    ir107_norm = tf.keras.layers.Lambda(lambda x,mu,scale: (x-mu)/scale,
                                       arguments={'mu':norm['ir107']['shift'],'scale':norm['ir107']['scale']})(ir107)
    lght_norm = tf.keras.layers.Lambda(lambda x,mu,scale: (x-mu)/scale,
                                       arguments={'mu':norm['lght']['shift'],'scale': norm['lght']['scale']})(lght)

    # reshape lght into 192
    lght_res = tf.keras.layers.Lambda(lambda t:  tf.image.resize(t,(192,192)))(lght_norm)

    # concat all inputs
    x_inp = tf.keras.layers.Lambda(lambda xs: tf.concat((xs[0],xs[1],xs[2]) ,axis=-1))([ir069_norm,ir107_norm,lght_res])
    
    encoder0_pool, encoder0 = encoder_block(x_inp, start_filters)
    encoder1_pool, encoder1 = encoder_block(encoder0_pool, start_filters*2)
    encoder2_pool, encoder2 = encoder_block(encoder1_pool, start_filters*4)
    encoder3_pool, encoder3 = encoder_block(encoder2_pool, start_filters*8)
    center = conv_block(encoder3_pool, start_filters*32)
    decoder3 = decoder_block(center, encoder3, start_filters*8)
    decoder2 = decoder_block(decoder3, encoder2, start_filters*6)
    decoder1 = decoder_block(decoder2, encoder1, start_filters*4)
    decoder0 = decoder_block(decoder1, encoder0, start_filters*2)
    # need one more upsample to get the resolution right
    decoder00 = decoder_block(decoder0, None, start_filters)
    output = layers.Conv2D(1, (1, 1), padding='same', activation='linear', name='output_layer')(decoder00)
    
    model = models.Model(inputs=inputs, outputs=output, name='generator')
    return model



def downsample(filters, size, apply_batchnorm=True):
    initializer = tf.random_normal_initializer(0., 0.02)

    result = tf.keras.Sequential()
    result.add(
      tf.keras.layers.Conv2D(filters, size, strides=2, padding='same',
                             kernel_initializer=initializer, use_bias=False))

    if apply_batchnorm:
        result.add(tf.keras.layers.BatchNormalization())

    result.add(tf.keras.layers.LeakyReLU())

    return result


# discriminator and downsample code from :
# https://colab.research.google.com/github/tensorflow/docs/blob/master/site/en/tutorials/generative/pix2pix.ipynb
# discriminator is a patchGAN - basically a convnet with B X H x W x C output
# instead of a single true/false output
def discriminator(input_shape=(384,384,1), target_input_shape=(384,384,1)):
    initializer = tf.random_normal_initializer(0., 0.02)

    #inp = layers.Input(shape=input_shape, name='input_image') # this is the generator output
    
    ir069 = tf.keras.layers.Input(shape=(192,192,1)) 
    ir107 = tf.keras.layers.Input(shape=(192,192,1)) 
    lght  = tf.keras.layers.Input(shape=(48,48,1))
    
    # reshape everything to target_input_shape
    ir069_res = tf.keras.layers.Lambda(lambda t:  tf.image.resize(t,(384,384)))(ir069)
    ir107_res = tf.keras.layers.Lambda(lambda t:  tf.image.resize(t,(384,384)))(ir107)
    lght_res = tf.keras.layers.Lambda(lambda t:  tf.image.resize(t,(384,384)))(lght)
    
    tar = layers.Input(shape=target_input_shape, name='target_image') # this is the ground truth

    x = layers.concatenate([ir069_res,ir107_res,lght_res,tar])

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

    model = models.Model(inputs=[ir069,ir107,lght,tar], outputs=output)
    return model

