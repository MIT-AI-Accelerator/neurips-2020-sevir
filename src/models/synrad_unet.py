"""
unet definition for synthetic weather radar using SEVIR
"""

import tensorflow as tf
from tensorflow.keras import layers,losses,models,regularizers
from .unet import conv_block,encoder_block,decoder_block

def create_model(norm, start_filters=32):
    
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
    
    return inputs, output









