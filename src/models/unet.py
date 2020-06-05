"""
block definitions for unet model
"""

import tensorflow as tf
from tensorflow.keras import layers,losses,models,regularizers

def conv_block(input_tensor, num_filters, activation='relu'):
    encoder = layers.Conv2D(num_filters, (3, 3), padding='same')(input_tensor)
    encoder = layers.BatchNormalization()(encoder)
    encoder = layers.Activation(activation)(encoder)
    encoder = layers.Conv2D(num_filters, (3, 3), padding='same')(encoder)
    encoder = layers.BatchNormalization()(encoder)
    encoder = layers.Activation(activation)(encoder)
    return encoder

def encoder_block(input_tensor, num_filters, activation='relu', resnet_style=False):
    encoder = conv_block(input_tensor, num_filters, activation)
    if resnet_style:
        # add the cross/resnet style connection by concatenating input to the encoder output
        encoder = layers.concatenate([encoder, input_tensor], axis=-1)
    encoder_pool = layers.MaxPooling2D((2, 2), strides=(2, 2))(encoder)
    return encoder_pool, encoder

def decoder_block(input_tensor, concat_tensor, num_filters, activation='relu'):
    decoder = layers.Conv2DTranspose(num_filters, (2, 2), strides=(2, 2), padding='same')(input_tensor)
    if concat_tensor is not None:
        decoder = layers.concatenate([concat_tensor, decoder], axis=-1)
    decoder = layers.BatchNormalization()(decoder)
    decoder = layers.Activation(activation)(decoder)
    decoder = layers.Conv2D(num_filters, (3, 3), padding='same')(decoder)
    decoder = layers.BatchNormalization()(decoder)
    decoder = layers.Activation(activation)(decoder)
    decoder = layers.Conv2D(num_filters, (3, 3), padding='same')(decoder)
    decoder = layers.BatchNormalization()(decoder)
    decoder = layers.Activation(activation)(decoder)
    return decoder

