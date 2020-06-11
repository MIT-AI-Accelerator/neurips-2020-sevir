# Based on UNet implementation here :
# https://colab.research.google.com/github/MarkDaoust/models/blob/segmentation_blogpost/samples/outreach/blogs/segmentation_blogpost/image_segmentation.ipynb

import tensorflow as tf
from tensorflow.keras import layers,losses,models,regularizers
from .unet import conv_block,encoder_block,decoder_block

def create_model(input_shape=(384,384,13), num_outputs=12, activation='relu'):
    inputs = layers.Input(shape=input_shape)
    encoder0_pool, encoder0 = encoder_block(inputs, 32, activation=activation)
    encoder1_pool, encoder1 = encoder_block(encoder0_pool, 64, activation=activation)
    encoder2_pool, encoder2 = encoder_block(encoder1_pool, 128, activation=activation)
    encoder3_pool, encoder3 = encoder_block(encoder2_pool, 256, activation=activation)
    center = conv_block(encoder3_pool, 1024)
    decoder3 = decoder_block(center, encoder3, 256)
    decoder2 = decoder_block(decoder3, encoder2, 128, activation=activation)
    decoder1 = decoder_block(decoder2, encoder1, 64, activation=activation)
    decoder0 = decoder_block(decoder1, encoder0, 32, activation=activation)
    outputs = layers.Conv2D(num_outputs, (1, 1), padding='same',
                                activation='linear', name='output_layer')(decoder0)
    
    return inputs, outputs

