import sys
sys.path.append('src/')

import os
import time
import datetime
import logging
import argparse
import numpy as np
from socket import gethostname
import csv

import tensorflow as tf
from tensorflow.keras import models,optimizers
from tensorflow.keras.losses import mean_squared_error
from tensorflow.keras.callbacks import ModelCheckpoint, LambdaCallback
from models.synrad_unet import create_model
from losses.style_loss import vggloss,vggloss_scaled
from readers.synrad_reader import get_data
from metrics import probability_of_detection,success_rate,critical_success_index
from losses.vggloss import VGGLoss # VGG 19 content loss
from utils.trainutils import train_step
from readers.normalizations import zscore_normalizations as NORM
from utils.utils import setuplogging,log_args,print_args,make_callback_dirs
from models.synrad_gan import generator,discriminator
from losses.gan_losses import generator_loss,discriminator_loss


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--loss_fn', type=str, default='mse', choices=['mse', 'mse+vgg','gan+mae'])
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
    logging.info(f'Creating directories')
    tensorboard_dir,imgs_dir,weights_dir = make_callback_dirs(logdir)
    metrics_file = os.path.join(logdir, 'metrics.csv')

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
    opt = tf.keras.optimizers.Adam(lr=float(args.lr))
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

    types  = ((tf.float32,tf.float32,tf.float32),  len(loss_fn)*(tf.float32,) )
    shapes = ( ((None,192,192,1),(None,192,192,1),(None,48,48,1)), len(loss_fn)*((None,384,384,1),) ) 
    train_dataset = tf.data.Dataset.from_generator(lambda : train_gen, output_types=types, output_shapes=shapes)
    val_dataset   = tf.data.Dataset.from_generator(lambda : val_gen, output_types=types, output_shapes=shapes)

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




def train_gan(data,args):
    """
    main for GAN case
    """
    global_rank=0 # we are assuming 1 GPU
    num_data = data[1]['vil'].shape[0]
    
    tensorboard_dir = os.path.join(args.logdir, 'tensorboard')
    images_dir = os.path.join(args.logdir, 'images')
    weights_dir = os.path.join(args.logdir, 'weights')
    summary_file = os.path.join(tensorboard_dir, datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

    # only rank 0 should do this
    if global_rank == 0:
        logging.info(f'rank {global_rank} creating directories')
        make_callback_dirs(tensorboard_dir,images_dir,weights_dir)       
        summary_writer = tf.summary.create_file_writer(summary_file)
        # only rank 0 will write the metrics file
        csv_file = open(os.path.join(args.logdir, 'metrics.csv'), 'w')
        writer = csv.writer(csv_file)
        writer.writerow(['gen_total_loss','gen_gan_loss','gen_l1_loss','disc_loss'])
        # read data for visualizing intermediate outputs
        #viz_IN, viz_OUT = get_viz_inputs()
    else:
        viz_IN = None
        viz_OUT = None
        summary_writer = None
    
    # instantiate models
    g = generator(NORM)
    d = discriminator()

    # optimizers
    g_opt = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
    d_opt = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
    
    logging.info('training')
    start = time.time()
    for epoch in range(args.nepochs):

        t0 = time.time()
        # iterate over the data in batches
        for idx in range(0, num_data, args.batch_size):
            # take a batch of data and train
            #input_image = data[0][idx:idx+args.batch_size,::]
            input_image = [data[0]['ir069'][idx:idx+args.batch_size,::],
                           data[0]['ir107'][idx:idx+args.batch_size,::],
                           data[0]['lght'][idx:idx+args.batch_size,::]]
            target = data[1]['vil'][idx:idx+args.batch_size,::]
            losses = train_step(g, g_opt, d, d_opt, input_image, target, epoch, summary_writer)

        # write stats at the end of the epoch
        if global_rank==0:
            #gen_total_loss,gen_gan_loss,gen_l1_loss,disc_loss
            losses = [L.numpy() for L in losses]
            writer.writerow(losses)
            csv_file.flush() # to ensure that each line is written as we go along

            # since we write our own training loop, there is no concept of callbacks
            # this is the end of one epoch - here we can run on the test/val data and generate
            g.save(os.path.join(weights_dir, f'trained_generator-{epoch}.h5'))
            d.save(os.path.join(weights_dir, f'trained_discriminator-{epoch}.h5'))

        T = time.time()-t0
        logging.info(f'epoch {epoch} - {T} sec. - gen_loss : {losses[0]} - disc_loss : {losses[-1]}')
    
    logging.info('training time : {time.time()-start sec.}')
    if global_rank==0:
        csv_file.close()
    return


def main(args):
    data      = get_data(args.train_data, end=args.num_train )

    if args.loss_fn in ['gan+mae']: # gan case has custom train loop
        train_gan(data,args)
    else: # non-gan case
        model,loss_fn,loss_weights = get_model(args)
        callbacks = get_callbacks(model, args.logdir)
        train(model, data, args.batch_size, args.nepochs, 
              loss_fn, loss_weights, callbacks, args.verbosity) 
    
    return


if __name__ == '__main__':
    args = get_args()
    if not os.path.exists(args.logdir):
        os.mkdir(args.logdir)
    setuplogging(os.path.join(args.logdir, f'0.log'))
    log_args(args)
    logging.info(f'Host: {gethostname()}')
    main(args)    
    print('all done')





