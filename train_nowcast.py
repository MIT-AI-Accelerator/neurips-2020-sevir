import os
import sys
sys.path.append(os.path.join(os.getcwd(), 'src'))

import time
import logging
import pandas as pd
import numpy as np
from socket import gethostname
import tensorflow as tf
from tensorflow.keras import models,optimizers
from tensorflow.keras.losses import mean_squared_error
from tensorflow.keras.callbacks import ModelCheckpoint,LambdaCallback,TensorBoard,CSVLogger
from models.nowcast_unet import create_model
from losses.style_loss import vggloss,vggloss_scaled
from cbhelpers.custom_callbacks import make_callback_dirs
from readers.nowcast_reader import get_data
from utils.utils import setuplogging,log_args,print_args
from metrics import probability_of_detection,success_rate,critical_success_index

MEAN=33.44
SCALE=47.54

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--loss_fn', type=str, default='mse', choices=['mse', 'vgg', 'mse+vgg'])
    parser.add_argument('--loss_weights', nargs='+', default="1.0")
    parser.add_argument('--train_data', type=str, help='path to training data file',default='data/interim/nowcast_training.h5')
    parser.add_argument('--nepochs', type=int, help='number of epochs', default=5)    
    parser.add_argument('--batch_size', type=int, help='batch size', default=32)
    parser.add_argument('--num_train', type=int, help='number of training sequences to read', default=None)
    parser.add_argument('--lr', type=float, help='learning rate', default=0.001)
    parser.add_argument('--verbosity', type=int, default=2)
    parser.add_argument('--logdir', type=str, help='log directory', default='./logs')
    args, unknown = parser.parse_known_args()

    if args.loss_fn.find('vgg')>0:
        # a smaller batch size is needed when using vgg style and content loss
        args.batch_size = 4
    return args

def get_loss_fn(loss):
    choices = {'mse':[mean_squared_error],
                   'vgg':[vggloss],
                   'mse+vgg':[mean_squared_error, vggloss_scaled] }
    return choices[loss]

def get_callbacks(model, logdir, num_warmup=1):
    # define callback directories
    tensorboard_dir = os.path.join(logdir, 'tensorboard')
    imgs_dir = os.path.join(logdir, 'images')
    weights_dir = os.path.join(logdir, 'weights')

    logging.info(f'tensorboard dir : {tensorboard_dir}')
    logging.info(f'image dir : {imgs_dir}')
    logging.info(f'weights dir : {weights_dir}')

    logging.info(f'creating directories')
    tensorboard_dir,imgs_dir,weights_dir = make_callback_dirs(tensorboard_dir,imgs_dir,weights_dir)

    # define callbacks
    metrics_file = os.path.join(logdir, 'metrics.csv')
    save_checkpoint = ModelCheckpoint(os.path.join(weights_dir, 'weights_{epoch:02d}.h5'),
                                          save_best=True, save_weights_only=False, mode='auto')

    tensorboard_callback = TensorBoard(log_dir=tensorboard_dir)    
    csv_callback = CSVLogger( metrics_file ) 
    callbacks += [csv_callback, save_checkpoint, tensorboard_callback]

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
    loss_weights = [np.float32(W) for W in args.loss_weights]

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

def main(args):
    model,loss_fn,loss_weights = get_model(args)
    callbacks = get_callbacks(model, args.logdir, args.num_warmup)
    data = get_data(args.train_data, args.test_data, num_train=args.num_train, pct_validation=args.num_test)

    logging.info('data loading completed')
    logging.info(f'num. train : {data[0].shape}')
    logging.info(f'num. val   : {data[2].shape}')

    y = train(model, data, args.batch_size, args.nepochs, loss_fn, loss_weights, callbacks, args.verbosity)

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

    
