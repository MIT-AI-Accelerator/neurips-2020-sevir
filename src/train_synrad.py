# horovod initialization must happen globally
# before anything else - similar to MPI
from utils.distutils import setup_gpu_mapping
import horovod.tensorflow.keras as hvd
# initialize horovod first
hvd.init()
# assign unique gpu per rank
setup_gpu_mapping(hvd)

import os
import time
import logging
import numpy as np
from socket import gethostname
import tensorflow as tf
from tensorflow.keras import models,optimizers
from tensorflow.keras.losses import mean_squared_error
from tensorflow.keras.callbacks import ModelCheckpoint, LambdaCallback
from models.synrad_unet import create_model
from losses.style_loss import vggloss,vggloss_scaled
#from losses.vil_loss import vil_loss,vil_loss_multilevel
from cbhelpers.custom_callbacks import save_predictions,make_callback_dirs
from readers.synrad_reader import read_data
from metrics import probability_of_detection,success_rate,critical_success_index

from losses.vggloss import VGGLoss

from horovod.tensorflow.keras import DistributedOptimizer

from readers.normalizations import zscore_normalizations as NORM
from utils.utils import default_args,setuplogging,log_args,print_args

RANDOM_STATE = 1234


def get_data(train_data, test_data, pct_validation=0.2 ):
    logging.info('Reading datasets')
    train_IN, train_OUT = read_data(train_data, rank=hvd.rank(), size=hvd.size(),end=None)
    #test_IN, test_OUT   = read_data(test_data, rank=hvd.rank(), size=hvd.size(),end=500)

    # Make the validation dataset the last pct_validation of the training data
    val_idx = int((1-pct_validation)*len(train_OUT['vil']))
    val_IN={}
    val_OUT={}
    for k in train_IN:
        train_IN[k],val_IN[k]=train_IN[k][:val_idx],train_IN[k][val_idx:]
    for k in train_OUT:
        train_OUT[k],val_OUT[k]=train_OUT[k][:val_idx],train_OUT[k][val_idx:]
    
    # shuffle the datasets (in unison)
    #logging.info('Shuffling datasets')
    #rs = np.random.get_state()
    #for k in train_IN:
    #    np.random.set_state(rs)
    #    np.random.shuffle(train_IN[k])
    #    np.random.set_state(rs)
    #    np.random.shuffle(val_IN[k])
    #for k in train_IN:
    #    np.random.set_state(rs)
    #    np.random.shuffle(train_OUT[k])
    #    np.random.set_state(rs)
    #    np.random.shuffle(val_OUT[k])

    logging.info('data loading completed')
    return (train_IN,train_OUT,val_IN,val_OUT)

def get_callbacks(model, logdir, num_warmup=1):
    # define callback directories
    tensorboard_dir = os.path.join(logdir, 'tensorboard')
    imgs_dir = os.path.join(logdir, 'images')
    weights_dir = os.path.join(logdir, 'weights')
    metrics_file = os.path.join(logdir, 'train.log')

    logging.info(f'tensorboard dir : {tensorboard_dir}')
    logging.info(f'image dir : {imgs_dir}')
    logging.info(f'weights dir : {weights_dir}')

    # only rank 0 should do this
    if hvd.rank() == 0:
        logging.info(f'rank {hvd.rank()} creating directories')
        tensorboard_dir,imgs_dir,weights_dir = make_callback_dirs(tensorboard_dir,imgs_dir,weights_dir)

    # make all ranks wait to ensure log dirs are created
    logging.info(f'rank {hvd.rank()} waiting on barrier')
    hvd.allreduce([0], name="Barrier")

    # define callbacks
    callbacks = [ hvd.callbacks.BroadcastGlobalVariablesCallback(0),
                 hvd.callbacks.LearningRateWarmupCallback(warmup_epochs=num_warmup, verbose=2),
                 hvd.callbacks.MetricAverageCallback() ]

    if hvd.rank()==0:
        save_checkpoint = ModelCheckpoint(os.path.join(weights_dir, 'weights_{epoch:02d}.h5'),
                                          save_best=True, save_weights_only=False, mode='auto')
        
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=tensorboard_dir)   

        csvwriter = tf.keras.callbacks.CSVLogger( metrics_file ) 

        callbacks += [save_checkpoint, tensorboard_callback, csvwriter]

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
    
    vgg = VGGLoss(input_shape=(384,384,1),
                  resize_to=None,
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
    opt = optimizers.Adam()
    logging.info('compiling model')
    model.compile(optimizer=DistributedOptimizer(opt),
                      loss_weights=loss_weights,
                      metrics=get_metrics(),
                      loss=loss_fn,
                      experimental_run_tf_function=False)
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
                yield inputs_batch, outputs_batch
            else:
                yield inputs_batch

def train(model, data, batch_size, num_epochs, loss_fn, loss_weights, callbacks, verbosity):
    # data : (train_IN,train_OUT,test_IN,test_OUT)
    # train
    logging.info('training...')
    t0 = time.time()
    train_gen = generate(inputs=[data[0]['ir069'],data[0]['ir107'],data[0]['lght']],
                         outputs=len(loss_fn)*[data[1]['vil']],batch_size=batch_size)
    val_gen = generate(inputs=[data[2]['ir069'],data[2]['ir107'],data[2]['lght']],
                         outputs=len(loss_fn)*[data[3]['vil']],batch_size=batch_size)
    #y = model.fit(x=[data[0]['ir069'],data[0]['ir107'],data[0]['lght']], 
    #              y=len(loss_fn)*[data[1]['vil']],
    y = model.fit_generator(train_gen,
                              epochs=num_epochs,
                              steps_per_epoch=75,
                              validation_data=val_gen, # WONT WORK IN TF 2.1
                              validation_steps=75,
                             # validation_data=([data[2]['ir069'],data[2]['ir107'],data[2]['lght']], 
                             #                   len(loss_fn)*[data[3]['vil']]),
                              callbacks=callbacks,
                              verbose=verbosity)
    t1 = time.time()
    logging.info(f'training time : {t1-t0} sec.')
    
    return

def main(args):
    model,loss_fn,loss_weights = get_model(args)
    callbacks = get_callbacks(model, args.logdir, args.num_warmup)
    data      = get_data(args.train_data, args.test_data )
    train(model, data, args.batch_size, args.nepochs, loss_fn, loss_weights, callbacks, args.verbosity) 
    return

if __name__ == '__main__':
    args = default_args()
    # logging has to be setup after horovod init
    setuplogging(os.path.join(args.logdir, f'{hvd.rank()}.log'))
    log_args(args)
    logging.info(f'MPI size : {hvd.size()}, {gethostname()},  my rank : {hvd.rank()}')
    main(args)    
    print('all done')

    
