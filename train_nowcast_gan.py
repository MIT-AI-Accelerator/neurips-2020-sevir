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
from readers.nowcast_reader import read_data
from models.nowcast_gan import generator,discriminator
from losses.gan_losses import generator_loss,discriminator_loss
from cbhelpers.custom_callbacks import save_predictions,make_callback_dirs,get_viz_inputs
from utils.trainutils import train_step
from utils.utils import default_args,setuplogging,log_args,print_args

def get_data(train_data, test_data, num_train=None, pct_validation=0.2):
    # read data: this function returns scaled data
    # what about shuffling ? 
    logging.info(f'rank {global_rank} reading images')
    train_IN, train_OUT = read_data(train_data,
                                        rank=global_rank,
                                        size=global_size,
                                        end=num_train)
    # change the following :
    # how many test sets should be read by each rank ?
    # offset the test data reading based on rank and number of clips to be read

    # Make the validation dataset the last pct_validation of the training data
    if pct_validation<1:
        val_idx = int((1-pct_validation)*train_IN.shape[0])
    else:
        val_idx = train_IN.shape[0]-pct_validation
    
    val_IN = train_IN[val_idx:, ::]
    train_IN = train_IN[:val_idx, ::]

    val_OUT = train_OUT[val_idx:, ::]
    train_OUT = train_OUT[:val_idx, ::]

    logging.info('data loading completed')
    logging.info(f'rank {global_rank} train : {train_IN.shape}')
    logging.info(f'rank {global_rank} val   : {val_IN.shape}')
    return (train_IN,train_OUT,val_IN,val_OUT)

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
    # This only changes the training function used.
    # The code is still launched with MPI and horovod - this needs to be changed
    # eventually
    data = get_data(args.train_data, args.test_data, num_train=args.num_train, pct_validation=args.num_test)
    num_data = data[0].shape[0]

    # read data for visualization during training in save_prediction callback
    viz_IN, viz_OUT = get_viz_inputs()
    tensorboard_dir = os.path.join(args.logdir, 'tensorboard')
    images_dir = os.path.join(args.logdir, 'images')
    weights_dir = os.path.join(args.logdir, 'weights')
    summary_file = os.path.join(tensorboard_dir, datetime.now().strftime("%Y%m%d-%H%M%S"))
    
    logging.info(f'tensorboard dir : {tensorboard_dir}')
    logging.info(f'image dir : {images_dir}')
    logging.info(f'weights dir : {weights_dir}')
    logging.info(f'tensorflow summary file : {summary_file}')

    # only rank 0 should do this
    if global_rank == 0:
        logging.info(f'rank {global_rank} creating directories')
        make_callback_dirs(tensorboard_dir,images_dir,weights_dir)       
        summary_writer = tf.summary.create_file_writer(summary_file)
    else:
        summary_writer = None
    
    # instantiate models
    g = generator()
    d = discriminator()

    # optimizers
    g_opt = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
    d_opt = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
  
    start = time.time()
    ii = 0
    logging.info('training')
    with open(os.path.join(args.logdir, 'metrics.csv'), 'w') as fp:
        writer = csv.writer(fp)
        writer.writerow(['gen_total_loss','gen_gan_loss','gen_l1_loss','disc_loss'])
        for epoch in range(args.nepochs):
            t0 = time.time()
            for idx in range(0, num_data, args.batch_size):
                # take a batch of data and train
                input_image = data[0][idx:idx+args.batch_size,::]
                target = data[1][idx:idx+args.batch_size,::]
                gen_total_loss,gen_gan_loss,gen_l1_loss,disc_loss = train_step(g, g_opt,
                                                                                 d, d_opt,
                                                                                 input_image,
                                                                                 target, epoch,
                                                                                 summary_writer)
            # write stats at the end of the batch
            # need to add metrics here
            writer.writerow([gen_total_loss.numpy(),gen_gan_loss.numpy(),
                                 gen_l1_loss.numpy(),disc_loss.numpy()])
            fp.flush() # to ensure that each line is written as we go along
            
            logging.info(f'epoch {epoch} - {time.time()-t0} sec. - disc_loss : {disc_loss}')

            # since we write our own training loop, there is no concept of callbacks
            # this is the end of one epoch - here we can run on the test/val data and generate
            # intermediate outputs
            save_predictions(epoch, images_dir, 74., g, viz_IN, viz_OUT, MEAN=33.44, SCALE=47.54)
            g.save(os.path.join(weights_dir, f'trained_generator-{epoch}.h5'))
            d.save(os.path.join(weights_dir, f'trained_discriminator-{epoch}.h5'))
    logging.info('training time : {time.time()-start sec.}')
    return

if __name__ == '__main__':
    args = default_args()
    # logging has to be setup after horovod init
    setuplogging(os.path.join(args.logdir, f'{global_rank}.log'))    
    log_args(args)
    logging.info(f'MPI size : {global_size}, {gethostname()},  my rank : {global_rank}')
    main(args)    
    print('all done')

    
    
