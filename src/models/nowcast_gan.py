import tensorflow as tf
from tensorflow.keras import layers,losses,models
from losses.gan_losses import generator_loss,discriminator_loss
from .unet_gan import conv_block,encoder_block,decoder_block,downsample

def generator(input_shape=(384,384,13), num_outputs=12):
    inputs = layers.Input(shape=input_shape)
    encoder0_pool, encoder0 = encoder_block(inputs, 32)
    encoder1_pool, encoder1 = encoder_block(encoder0_pool, 64)
    encoder2_pool, encoder2 = encoder_block(encoder1_pool, 128)
    encoder3_pool, encoder3 = encoder_block(encoder2_pool, 256)
    center = conv_block(encoder3_pool, 1024)
    decoder3 = decoder_block(center, encoder3, 256)
    decoder2 = decoder_block(decoder3, encoder2, 128)
    decoder1 = decoder_block(decoder2, encoder1, 64)
    decoder0 = decoder_block(decoder1, encoder0, 32)
    outputs = layers.Conv2D(num_outputs, (1, 1), padding='same',
                                activation='linear', name='output_layer')(decoder0)
    
    model = models.Model(inputs=inputs, outputs=outputs, name='generator')
    return model

# discriminator and downsample code from :
# https://colab.research.google.com/github/tensorflow/docs/blob/master/site/en/tutorials/generative/pix2pix.ipynb
# discriminator is a patchGAN - basically a convnet with B X H x W x C output
# instead of a single true/false output
def discriminator(input_shape=(384,384,13), target_input_shape=(384,384,12)):
    initializer = tf.random_normal_initializer(0., 0.02)

    inp = layers.Input(shape=input_shape, name='input_image') # this is the generator output
    tar = layers.Input(shape=target_input_shape, name='target_image') # this is the ground truth

    x = layers.concatenate([inp, tar])

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

    model = models.Model(inputs=[inp, tar], outputs=output)
    return model

