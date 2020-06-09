import os
import argparse
import logging

def setuplogging(logfile='0.log'):
    logging.basicConfig(filename=logfile,
                            level=logging.INFO,
                            filemode='w',
                            format='%(asctime)s - %(message)s')
    return

def setupmetricslog(logger_name='metricslogger', logfile='metrics.csv'):
    logger = logging.getLogger(logger_name)
    formatter = logging.Formatter('%(message)s')
    fh = logging.FileHandler(logfile, mode='w')
    fh.setFormatter(formatter)
    logger.setLevel(logging.INFO)
    logger.addHandler(fh)
    return logger

def setuprootlogger(logger_name='root', logfile='0.log'):
    logger = logging.getLogger(logger_name)
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    fh = logging.FileHandler(logfile, mode='w')
    fh.setFormatter(formatter)
    logger.setLevel(logging.INFO)
    logger.addHandler(fh)
    return logger

def print_args(args):
    for arg in vars(args):
        print("{0: <15} : {1:}".format(arg, getattr(args, arg)))
    return

def log_args(args, logger_name=None):
    if logger_name is None:
        logger = logging.getLogger() # get the default logger
    else:
        logger = logger.getLogger(logger_name)
    for arg in vars(args):
        logger.info("{0: <15} : {1:}".format(arg, getattr(args, arg)))
    return

def default_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='unet',
                            help='name of model to train',
                            choices=['unet', 'nowcast', 'synrad', 'nowcast_gan'])
    parser.add_argument('--loss_fn', type=str, default='mse',
                            choices=['mse', 'vil', 'multilevel-vil', 'vgg', 'myloss', 'ce', 'ssim',
                                         'mse+vil', 'mse+multilevel-vil', 'mse+vgg', 'multiscale',
                                         'mse+ssim', 'featureloss', 'gan', 'binary_crossentropy'])
    parser.add_argument('--loss_weights', nargs='+', default="[1.0]")
    parser.add_argument('--train_data', type=str, help='path to training data file')
    parser.add_argument('--test_data', type=str, help='path to test data file')
    parser.add_argument('--nepochs', type=int, help='number of epochs', default=50)    
    parser.add_argument('--batch_size', type=int, help='batch size', default=8)
    parser.add_argument('--num_train', type=int, help='number of training sequences to read',
                            default=2048)
    parser.add_argument('--test_start', type=int, help='where to start reading test data from',
                            default=4096)
    parser.add_argument('--num_test', type=float, help='number of test sequences to read',
                            default=8)
    parser.add_argument('--lr', type=float, help='learning rate', default=0.000001)
    parser.add_argument('--verbosity', type=int, default=2)
    parser.add_argument('--no_hvd', action='store_true', help='disable Horovod')
    parser.add_argument('--logdir', type=str, help='log directory', default='./logs')
    parser.add_argument('--num_warmup', type=int, help='number of warmup epochs', default=2)
    parser.add_argument('--gen_weights', type=str, help='path to pre-trained unet/generator weights',
                            default=None)
    parser.add_argument('--disc_weights', type=str, help='path to pre-trained discriminator weights',
                            default=None)
    args, unknown = parser.parse_known_args()

    return args
