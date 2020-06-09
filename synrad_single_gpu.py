import sys
sys.path.append('src/')

import os
import time
import logging
import argparse
import numpy as np
from socket import gethostname

import tensorflow as tf
from tensorflow.keras import models,optimizers
from tensorflow.keras.losses import mean_squared_error
from tensorflow.keras.callbacks import ModelCheckpoint, LambdaCallback
from models.synrad_unet import create_model
from losses.style_loss import vggloss,vggloss_scaled
from cbhelpers.custom_callbacks import save_predictions,make_callback_dirs
from readers.synrad_reader import get_data
from metrics import probability_of_detection,success_rate,critical_success_index
from losses.vggloss import VGGLoss # VGG 19 content loss

from readers.normalizations import zscore_normalizations as NORM
from utils.utils import default_args,setuplogging,log_args,print_args


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--loss_fn', type=str, default='mse', choices=['mse', 'mse+vgg'])
    parser.add_argument('--loss_weights', nargs='+', default="1.0")
    parser.add_argument('--train_data', type=str, help='path to training data file',default='data/interim/synrad_training.h5')
    parser.add_argument('--nepochs', type=int, help='number of epochs', default=5)    
    parser.add_argument('--batch_size', type=int, help='batch size', default=32)
    parser.add_argument('--num_train', type=int, help='number of training sequences to read', default=None)
    parser.add_argument('--lr', type=float, help='learning rate', default=0.001)
    parser.add_argument('--verbosity', type=int, default=2)
    parser.add_argument('--logdir', type=str, help='log directory', default='./logs')
    args, unknown = parser.parse_known_args()

    return args



def get_callbacks(model, logdir ):
    # define callback directories
    tensorboard_dir = os.path.join(logdir, 'tensorboard')
    imgs_dir = os.path.join(logdir, 'images')
    weights_dir = os.path.join(logdir, 'weights')
    metrics_file = os.path.join(logdir, 'train.log')

    logging.info(f'tensorboard dir : {tensorboard_dir}')
    logging.info(f'image dir : {imgs_dir}')
    logging.info(f'weights dir : {weights_dir}')

    # only rank 0 should do this
    logging.info(f'Creating directories')
    tensorboard_dir,imgs_dir,weights_dir = make_callback_dirs(tensorboard_dir,imgs_dir,weights_dir)

    save_checkpoint = ModelCheckpoint(os.path.join(weights_dir, 'weights_{epoch:02d}.h5'),
                                      save_best=True, save_weights_only=False, mode='auto')
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=tensorboard_dir)   
    csvwriter = tf.keras.callbacks.CSVLogger( metrics_file ) 
    callbacks = [save_checkpoint, tensorboard_callback, csvwriter]

    return callbacks


def get_metrics():
    def pod74(y_true,y_pred):
        return probability_of_detection(y_true,y_pred,np.array([74.0],dtype=np.float32))
    def pod133(y_true,y_pred):
        return probability_of_detection(y_true,y_pred,np.array([133.0],dtype=np.float32))
    def sucr74(y_true,y_pred):
        return success_rate(y_true,y_pred,np.array([74],dtype=np.float32))
    def sucr133(y_true,y_pred):
        return success_rate(y_true,y_pred,np.array([133],dtype=np.float32))
    def csi74(y_true,y_pred):
        return critical_success_index(y_true,y_pred,np.array([74],dtype=np.float32))
    def csi133(y_true,y_pred):
        return critical_success_index(y_true,y_pred,np.array([133],dtype=np.float32))
    return [pod74,pod133,sucr74,sucr133,csi74,csi133]


def get_loss_fn(loss):
    
    def synrad_mse(y_true,y_pred):
      return mean_squared_error(y_true,y_pred)/(NORM['vil']['scale']*NORM['vil']['scale'])
    
    vgg = VGGLoss(input_shape=(384,384,1), resize_to=None,
                  normalization_scale=np.float32(1.0), 
                  normalization_shift=-np.float32(NORM['vil']['shift']))
    vgg_loss = vgg.get_loss()

    def synrad_vgg(y_true,y_pred):
        return vgg_loss(y_true, y_pred)
    
    choices = {'mse':[synrad_mse],
               'mse+vgg':[synrad_mse, synrad_vgg]}
    
    return choices[loss]


def get_model(args):
    # define loss functions and weights
    loss_fn = get_loss_fn(args.loss_fn)
    loss_weights = [np.float32(W) for W in args.loss_weights]

    # create one or multi-output model based on the number of loss
    # functions provided
    inputs, outputs = create_model(NORM)
    model = models.Model(inputs=inputs, outputs=len(loss_fn)*[outputs])

    # create optimizer
    opt = tf.keras.optimizers.Adam()
    logging.info('compiling model')
    model.compile(optimizer=opt,
                  loss_weights=loss_weights,
                  metrics=get_metrics(),
                  loss=loss_fn)
    logging.info('compilation done')
    
    return model,loss_fn,loss_weights


def generate(inputs,outputs=None,batch_size=32, shuffle=True):
    """
    Simple generator for numpy arrays.  This will continue cycling through the datasets
    inputs  - list of input numpy arrays (matching first axis)
    outputs - list of output numpy arrays (matching first axis) 
    """
    N = inputs[0].shape[0]
    while True:
        idx = np.arange(N)
        if shuffle:
            np.random.shuffle(idx)
        batch_size = np.min([batch_size,len(idx)])
        imax = int(len(idx)/batch_size)
        for i in range(imax):
            idx_batch = idx[i*batch_size:(i+1)*batch_size]
            inputs_batch  = [X[idx_batch] for X in inputs]
            if outputs:
                outputs_batch = [Y[idx_batch] for Y in outputs]
                yield tuple(inputs_batch), tuple(outputs_batch)
            else:
                yield tuple(inputs_batch)


def train(model, data, batch_size, num_epochs, loss_fn, loss_weights, callbacks, verbosity):
    logging.info('training...')
    t0 = time.time()
    train_gen = generate(inputs=[data[0]['ir069'],data[0]['ir107'],data[0]['lght']],
                         outputs=len(loss_fn)*[data[1]['vil']],batch_size=batch_size)
    val_gen   = generate(inputs=[data[2]['ir069'],data[2]['ir107'],data[2]['lght']],
                         outputs=len(loss_fn)*[data[3]['vil']],batch_size=batch_size)
    
    train_dataset = tf.data.Dataset.from_generator(lambda : train_gen,
                                                   output_types=((tf.float32,tf.float32,tf.float32), 
                                                                 (tf.float32,) ),
                                                   output_shapes=( ((None,192,192,1),(None,192,192,1),(None,48,48,1)),
                                                                   ((None,384,384,1),) ))
    val_dataset   = tf.data.Dataset.from_generator(lambda : val_gen,
                                                   output_types=((tf.float32,tf.float32,tf.float32), 
                                                                 (tf.float32,) ),
                                                   output_shapes=( ((None,192,192,1),(None,192,192,1),(None,48,48,1)),
                                                                   ((None,384,384,1),) ))

    y = model.fit(train_dataset,
                  epochs=num_epochs,
                  steps_per_epoch=75,
                  validation_data=val_dataset, 
                  validation_steps=75,
                  callbacks=callbacks,
                  verbose=verbosity)
    t1 = time.time()
    logging.info(f'training time : {t1-t0} sec.')
    
    return


def main(args):
    model,loss_fn,loss_weights = get_model(args)
    callbacks = get_callbacks(model, args.logdir)
    data      = get_data(args.train_data, end=args.num_train )
    train(model, data, args.batch_size, args.nepochs, loss_fn, loss_weights, callbacks, args.verbosity) 
    return


if __name__ == '__main__':
    args = get_args()
    setuplogging(os.path.join(args.logdir, f'0.log'))
    log_args(args)
    logging.info(f'Host: {gethostname()}')
    main(args)    
    print('all done')





