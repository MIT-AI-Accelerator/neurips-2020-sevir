import os
import sys
sys.path.append(os.path.join(os.getcwd(), 'src'))

import csv
import time
import argparse
import logging
import pandas as pd
import numpy as np
import tensorflow as tf
from datetime import datetime
from socket import gethostname
from tensorflow.keras import models,optimizers
from tensorflow.keras.losses import mean_squared_error
from tensorflow.keras.callbacks import ModelCheckpoint,LambdaCallback,TensorBoard,CSVLogger
from models.nowcast_unet import create_model
from models.nowcast_gan import generator,discriminator
from losses.style_loss import vggloss,vggloss_scaled
from readers.nowcast_reader import get_data
from utils.trainutils import train_step
from utils.utils import setuplogging,log_args,print_args,make_callback_dirs
from metrics import probability_of_detection,success_rate,critical_success_index

MEAN=33.44
SCALE=47.54

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--loss_fn', type=str, default='mse', choices=['mse', 'vgg', 'mse+vgg', 'cgan'])
    parser.add_argument('--train_data', type=str, help='path to training data file',default='data/interim/nowcast_training.h5')
    parser.add_argument('--batch_size', type=int, help='batch size', default=4)
    parser.add_argument('--nepochs', type=int, help='number of epochs', default=5)    
    parser.add_argument('--num_train', type=int, help='number of training sequences to read', default=1024)
    parser.add_argument('--lr', type=float, help='learning rate', default=0.001)
    parser.add_argument('--verbosity', type=int, default=2)
    parser.add_argument('--logdir', type=str, help='log directory', default='./logs')
    args, unknown = parser.parse_known_args()

    return args

def get_loss_fn(loss):
    choices = {'mse':[mean_squared_error],
                   'vgg':[vggloss],
                   'mse+vgg':[mean_squared_error, vggloss_scaled] }
    return choices[loss]

def get_callbacks(model, logdir):
    # callback directories
    logging.info(f'creating directories')

    tensorboard_dir,imgs_dir,weights_dir = make_callback_dirs(logdir)

    logging.info(f'tensorboard dir : {tensorboard_dir}')
    logging.info(f'image dir : {imgs_dir}')
    logging.info(f'weights dir : {weights_dir}')

    # define callbacks
    metrics_file = os.path.join(logdir, 'metrics.csv')
    save_checkpoint = ModelCheckpoint(os.path.join(weights_dir, 'weights_{epoch:02d}.h5'),
                                          save_best=True, save_weights_only=False, mode='auto')

    tensorboard_callback = TensorBoard(log_dir=tensorboard_dir)    
    csv_callback = CSVLogger( metrics_file ) 
    callbacks = [csv_callback, save_checkpoint, tensorboard_callback]

    return callbacks

def get_metrics(nout=1):
    def pod74(y_true,y_pred):
        return probability_of_detection(y_true,y_pred,np.array(nout*[(74.0-MEAN)/SCALE],dtype=np.float32))
    def pod133(y_true,y_pred):
        return probability_of_detection(y_true,y_pred,np.array(nout*[(133.0-MEAN)/SCALE],dtype=np.float32))
    def sucr74(y_true,y_pred):
        return success_rate(y_true,y_pred,np.array(nout*[(74-MEAN)/SCALE],dtype=np.float32))
    def sucr133(y_true,y_pred):
        return success_rate(y_true,y_pred,np.array(nout*[(133-MEAN)/SCALE],dtype=np.float32))
    def csi74(y_true,y_pred):
        return critical_success_index(y_true,y_pred,np.array(nout*[(74-MEAN)/SCALE],dtype=np.float32))
    def csi133(y_true,y_pred):
        return critical_success_index(y_true,y_pred,np.array(nout*[(133-MEAN)/SCALE],dtype=np.float32))
    return [pod74,pod133,sucr74,sucr133,csi74,csi133]

def get_model(args):
    # define loss functions and weights
    loss_fn = get_loss_fn(args.loss_fn)
    # losses can potentially be weighted differently when
    # using MSE and VGG style and content as the loss function
    loss_weights = len(loss_fn)*[1.0]

    # create one or multi-output model based on the number of loss
    # functions provided
    inputs, outputs = create_model(input_shape=(384,384,13))
    model = models.Model(inputs=inputs, outputs=len(loss_fn)*[outputs])

    # create optimizer
    opt = optimizers.Adam()
    logging.info('compiling model')
    model.compile(optimizer=opt,
                      loss_weights=loss_weights,
                      metrics=get_metrics(outputs.get_shape()[-1]),
                      loss=loss_fn,
                      experimental_run_tf_function=False)
    logging.info('compilation done')
    
    return model,loss_fn,loss_weights

def train(model, data, batch_size, num_epochs, loss_fn, loss_weights, callbacks, verbosity):
    # data : (train_IN,train_OUT,test_IN,test_OUT)
    # train
    logging.info('training')
    t0 = time.time()
    y = model.fit(x=data[0], y=len(loss_fn)*[data[1]],
                  batch_size=batch_size,
                  epochs=num_epochs,
                  validation_data=(data[2], len(loss_fn)*[data[3]]),
                  callbacks=callbacks,
                  verbose=verbosity)
    t1 = time.time()
    logging.info(f'training time : {t1-t0} sec.')
    
    return y

def train_gan(data, args):
    # instantiate models
    g = generator()
    d = discriminator()

    # optimizers
    g_opt = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
    d_opt = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

    tensorboard_dir, _, weights_dir = make_callback_dirs(args.logdir)
    summary_file = os.path.join(tensorboard_dir, datetime.now().strftime("%Y%m%d-%H%M%S"))
    summary_writer = tf.summary.create_file_writer(summary_file)
    logging.info('training')
    with open(os.path.join(args.logdir, 'metrics.csv'), 'w') as fp:
        writer = csv.writer(fp)
        writer.writerow(['gen_total_loss','gen_gan_loss','gen_l1_loss','disc_loss'])
        for epoch in range(args.nepochs):
            for idx in range(0, data[0].shape[0], args.batch_size):
                # train on a batch of data 
                input_image = data[0][idx:idx+args.batch_size,::]
                target = data[1][idx:idx+args.batch_size,::]                
                gan_losses = train_step(g, g_opt, d, d_opt, [input_image], [target], epoch, summary_writer)
            
            # write stats at the end of the batch
            # gan_losses is a list of the following losses -
            # gen_total_loss,gen_gan_loss,gen_l1_loss,disc_loss
            writer.writerow([A.numpy() for A in gan_losses])
            fp.flush()            
            logging.info(f'epoch {epoch} - gen_total_loss: {gan_losses[0]} - gen_gan_loss: {gan_losses[1]} - gen_l1_loss: {gan_losses[2]} - disc_loss: {gan_losses[3]}')

            # since we write our own training loop, there is no concept of callbacks
            # this is the end of one epoch - here we can run on the test/val data and generate
            # intermediate outputs
            g.save(os.path.join(weights_dir, f'generator-{epoch}.h5'))
            d.save(os.path.join(weights_dir, f'discriminator-{epoch}.h5'))
    return

def main(args):
    data = get_data(args.train_data, end=args.num_train)

    logging.info('data loading completed')
    logging.info(f'num. train : {data[0].shape}')
    logging.info(f'num. val   : {data[2].shape}')

    if args.loss_fn == 'cgan':
        train_gan(data, args)
    else:
        model,loss_fn,loss_weights = get_model(args)
        callbacks = get_callbacks(model, args.logdir)
        y = train(model, data, args.batch_size, args.nepochs, loss_fn,
                      loss_weights, callbacks, args.verbosity)

        history_file = os.path.join(args.logdir, 'history.csv')
        hist_df = pd.DataFrame(y.history)
        with open(history_file, mode='w') as f:
            hist_df.to_csv(f)
        logging.info(f'history saved to {history_file}')

    return

if __name__ == '__main__':
    args = get_args()
    if not os.path.exists(args.logdir):
        os.mkdir(args.logdir)
    setuplogging(os.path.join(args.logdir, '0.log'))    
    log_args(args)
    main(args)    
    print('all done')

    
