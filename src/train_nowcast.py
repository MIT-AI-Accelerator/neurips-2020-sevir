# horovod initialization must happen globally
# before anything else - similar to MPI
from utils.distutils import setup_gpu_mapping
import horovod.tensorflow.keras as hvd
# initialize horovod first
hvd.init()
global_rank = hvd.rank()
global_size = hvd.size()
# assign unique gpu per rank
setup_gpu_mapping(hvd)

import os
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
from losses.vil_loss import vil_loss,vil_loss_multilevel
from losses.feature_loss import featureloss
from losses.experimental import myloss,multiscale,ce
from cbhelpers.custom_callbacks import save_predictions,make_callback_dirs,get_viz_inputs
#from readers.nowcast_reader import read_data
from readers.nowcast_reader import read_data_slow 
from utils.utils import default_args,setuplogging,log_args,print_args
from metrics import probability_of_detection,success_rate,critical_success_index
from horovod.tensorflow.keras import DistributedOptimizer

MEAN=33.44
SCALE=47.54

def get_data(train_data, test_data, num_train=None, pct_validation=0.2):
    # read data: this function returns scaled data
    # what about shuffling ? 
    logging.info(f'rank {global_rank} reading images')
    t0 = time.time()
    train_IN, train_OUT = read_data_slow(train_data,
                                            rank=global_rank,
                                            size=global_size,
                                             end=44760) # 44760 clips in training set
    #train_IN, train_OUT = read_data(train_data,
    #                                    rank=global_rank,
    #                                    size=global_size,
    #                                    end=num_train)
    t1 = time.time()
    logging.info(f'read time : {t1-t0}')
    # change the following :
    # how many test sets should be read by each rank ?
    # offset the test data reading based on rank and number of clips to be read

    # Make the validation dataset the last pct_validation of the training data
    if pct_validation<1:
        val_idx = int((1-pct_validation)*train_IN.shape[0])
    else:
        val_idx = int(train_IN.shape[0]-pct_validation)
    
    val_IN = train_IN[val_idx:, ::]
    train_IN = train_IN[:val_idx, ::]

    val_OUT = train_OUT[val_idx:, ::]
    train_OUT = train_OUT[:val_idx, ::]

    logging.info('data loading completed')
    logging.info(f'rank {global_rank} train : {train_IN.shape}')
    logging.info(f'rank {global_rank} val   : {val_IN.shape}')
    return (train_IN,train_OUT,val_IN,val_OUT)

def get_loss_fn(loss):
    choices = {'mse':[mean_squared_error],
                   'vil':[vil_loss],
                   'multilevel-vil':[vil_loss_multilevel],
                   'vgg':[vggloss],
                   'mse+vil':[mean_squared_error, vil_loss],
                   'mse+multilevel-vil':[mean_squared_error,vil_loss_multilevel],
                   'mse+vgg':[mean_squared_error, vggloss_scaled],
                   'myloss':[myloss],
                   'multiscale':[multiscale],
                   'featureloss':[featureloss],
                   'ce':[ce]}
    return choices[loss]

def get_callbacks(model, logdir, num_warmup=1):
    # define callback directories
    tensorboard_dir = os.path.join(logdir, 'tensorboard')
    imgs_dir = os.path.join(logdir, 'images')
    weights_dir = os.path.join(logdir, 'weights')

    logging.info(f'tensorboard dir : {tensorboard_dir}')
    logging.info(f'image dir : {imgs_dir}')
    logging.info(f'weights dir : {weights_dir}')

    # only rank 0 should do this
    if global_rank == 0:
        logging.info(f'rank {global_rank} creating directories')
        tensorboard_dir,imgs_dir,weights_dir = make_callback_dirs(tensorboard_dir,imgs_dir,weights_dir)

    # make all ranks wait to ensure log dirs are created
    logging.info(f'rank {global_rank} waiting on barrier')
    hvd.allreduce([0], name="Barrier")

    # define callbacks
    callbacks = [ hvd.callbacks.BroadcastGlobalVariablesCallback(0),
                      hvd.callbacks.LearningRateWarmupCallback(warmup_epochs=num_warmup, verbose=2) ]

    if global_rank==0:
        viz_IN, viz_OUT = get_viz_inputs()
        metrics_file = os.path.join(logdir, 'metrics.csv')
        save_checkpoint = ModelCheckpoint(os.path.join(weights_dir, 'weights_{epoch:02d}.h5'),
                                            save_best=True, save_weights_only=False, mode='auto')

        tensorboard_callback = TensorBoard(log_dir=tensorboard_dir)    
        predict_callback = LambdaCallback(on_epoch_end=lambda epoch,
                                          log:save_predictions(epoch, imgs_dir, 74.0,
                                                                   model, viz_IN, viz_OUT)) 
        csv_callback = CSVLogger( metrics_file ) 
        callbacks += [csv_callback, save_checkpoint, predict_callback, tensorboard_callback]

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
    model.compile(optimizer=DistributedOptimizer(opt),
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
    y = train(model, data, args.batch_size, args.nepochs, loss_fn, loss_weights, callbacks, args.verbosity)

    if global_rank==0:
        history_file = os.path.join(args.logdir, 'history.csv')
        hist_df = pd.DataFrame(y.history)
        with open(history_file, mode='w') as f:
            hist_df.to_csv(f)
        logging.info(f'history saved to {history_file}')

    return

if __name__ == '__main__':
    args = default_args()
    # logging has to be setup after horovod init
    setuplogging(os.path.join(args.logdir, f'{global_rank}.log'))
    log_args(args)
    logging.info(f'MPI size : {global_size}, {gethostname()},  my rank : {global_rank}')
    main(args)    
    print('all done')

    
