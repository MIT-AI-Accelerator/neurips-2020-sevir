# does distributed training change results ? 
# horovod initialization must happen globally
# before anything else - similar to MPI
from utils.distutils import setup_gpu_mapping
import horovod.tensorflow as hvd
# initialize horovod first
hvd.init()
global_rank = hvd.rank()
global_size = hvd.size()
# assign unique gpu per rank
setup_gpu_mapping(hvd)

import os
import csv
import time
import logging
import pandas as pd
import numpy as np
import tensorflow as tf
from datetime import datetime
from socket import gethostname
#from readers.nowcast_reader import read_data
from readers.nowcast_reader import read_data_slow
from models.nowcast_gan import generator,discriminator
from losses.gan_losses import generator_loss,discriminator_loss
from cbhelpers.custom_callbacks import save_predictions,make_callback_dirs,get_viz_inputs
from utils.trainutils import train_step_hvd
from utils.utils import default_args,setuplogging,log_args,print_args

MEAN=33.44
SCALE=47.54
def get_data(train_data, test_data, num_train=None, pct_validation=0.2):
    # read data: this function returns scaled data
    # what about shuffling ? 
    logging.info(f'rank {global_rank} reading images')
    train_IN, train_OUT = read_data_slow(train_data,
                                        rank=global_rank,
                                        size=global_size,
                                        end=44760) # 44760 clips in dataset
    #train_IN, train_OUT = read_data(train_data,
    #                                    rank=global_rank,
    #                                    size=global_size,
    #                                    end=num_train)
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

    #############
    # SET val_* to None since we are not using this data anyway
    val_IN = None
    val_OUT = None
    #############
    logging.info('data loading completed')
    if val_IN is not None:
        logging.info(f'rank {global_rank} train : {train_IN.shape}')
    if val_OUT is not None:
        logging.info(f'rank {global_rank} val   : {val_IN.shape}')
    return (train_IN,train_OUT,val_IN,val_OUT)

def get_models():
    # instantiate models
    g = generator()
    d = discriminator()

    # optimizers
    g_opt = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
    d_opt = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
    return g, g_opt, d, d_opt

def get_metrics(nout=1):
    def pod74(y_true,y_pred):
        return probability_of_detection(y_true,y_pred,np.array(nout*[74.0],dtype=np.float32))
    def pod133(y_true,y_pred):
        return probability_of_detection(y_true,y_pred,np.array(nout*[133.0],dtype=np.float32))
    def sucr74(y_true,y_pred):
        return success_rate(y_true,y_pred,np.array(nout*[74],dtype=np.float32))
    def sucr133(y_true,y_pred):
        return success_rate(y_true,y_pred,np.array(nout*[133],dtype=np.float32))
    def csi74(y_true,y_pred):
        return critical_success_index(y_true,y_pred,np.array(nout*[74],dtype=np.float32))
    def csi133(y_true,y_pred):
        return critical_success_index(y_true,y_pred,np.array(nout*[133],dtype=np.float32))
    return [pod74,pod133,sucr74,sucr133,csi74,csi133]

def main(args):
    # read trainign and validation data
    data = get_data(args.train_data, args.test_data, num_train=args.num_train, pct_validation=args.num_test)
    num_data = data[0].shape[0]
    
    tensorboard_dir = os.path.join(args.logdir, 'tensorboard')
    images_dir = os.path.join(args.logdir, 'images')
    weights_dir = os.path.join(args.logdir, 'weights')
    summary_file = os.path.join(tensorboard_dir, datetime.now().strftime("%Y%m%d-%H%M%S"))
    
    logging.info(f'tensorboard dir : {tensorboard_dir}')
    logging.info(f'image dir : {images_dir}')
    logging.info(f'weights dir : {weights_dir}')
    logging.info(f'tensorflow summary file : {summary_file}')
    metrics_fn = get_metrics(nout=12)
    
    # only rank 0 should do this
    if global_rank == 0:
        logging.info(f'rank {global_rank} creating directories')
        make_callback_dirs(tensorboard_dir,images_dir,weights_dir)       
        summary_writer = tf.summary.create_file_writer(summary_file)
        # only rank 0 will write the metrics file
        csv_file = open(os.path.join(args.logdir, 'metrics.csv'), 'w')
        writer = csv.writer(csv_file)
        writer.writerow(['gen_total_loss','gen_gan_loss','gen_l1_loss','disc_loss'])
        #,'pod74','pod133','sucr74','sucr133','csi74','csi133'])
        # read data for visualizing intermediate outputs
        viz_IN, viz_OUT = get_viz_inputs()
    else:
        viz_IN = None
        viz_OUT = None
        summary_writer = None
    
    g, g_opt, d, d_opt = get_models()
    
    logging.info('training')
    start = time.time()
    for epoch in range(args.nepochs):
        # make all ranks wait before starting an epoch
        logging.info(f'rank {global_rank} waiting on barrier')
        hvd.allreduce(tf.constant(0.))
        logging.info(f'rank {global_rank} starting training')

        t0 = time.time()
        # iterate over the data in batches
        for idx in range(0, num_data, args.batch_size):
            # take a batch of data and train
            input_image = data[0][idx:idx+args.batch_size,::]
            target = data[1][idx:idx+args.batch_size,::]
            losses = train_step_hvd(g, g_opt, d, d_opt, input_image, target, epoch, summary_writer, idx==0)

        # write stats at the end of the epoch
        if global_rank==0:
            #gen_total_loss,gen_gan_loss,gen_l1_loss,disc_loss
            losses = [L.numpy() for L in losses]

            # calculate metrics on val set
            #yp = g.predict(data[2], batch_size=4)
            #yp = yp*SCALE+MEAN
            #custom_metrics = [fn(yt*SCALE+MEAN,yp) for fn in metrics_fn]
            #writer.writerow(losses+custom_metrics)
            writer.writerow(losses)
            csv_file.flush() # to ensure that each line is written as we go along

            # since we write our own training loop, there is no concept of callbacks
            # this is the end of one epoch - here we can run on the test/val data and generate
            # intermediate outputs
            save_predictions(epoch, images_dir, 74., g, viz_IN, viz_OUT, MEAN=MEAN, SCALE=SCALE)
            g.save(os.path.join(weights_dir, f'trained_generator-{epoch}.h5'))
            d.save(os.path.join(weights_dir, f'trained_discriminator-{epoch}.h5'))

            
        T = time.time()-t0
        logging.info(f'epoch {epoch} - {T} sec. - gen_loss : {losses[0]} - disc_loss : {losses[-1]}')
    
    logging.info('training time : {time.time()-start sec.}')
    if global_rank==0:
        csv_file.close()
    return

if __name__ == '__main__':
    args = default_args()
    # logging has to be setup after horovod init
    setuplogging(os.path.join(args.logdir, f'{global_rank}.log'))    
    log_args(args)
    logging.info(f'MPI size : {hvd.size()}, {gethostname()},  my rank : {global_rank}')
    main(args)    
    print('all done')

    
    
