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

def make_callback_dirs(logdir):
    tensorboard_dir = os.path.join(logdir, 'tensorboard')
    if not os.path.isdir(tensorboard_dir):
        os.makedirs(tensorboard_dir)

    imgs_dir = os.path.join(logdir, 'images')
    if not os.path.isdir(imgs_dir):
        os.makedirs(imgs_dir)
    
    weights_dir = os.path.join(logdir, 'weights')
    if not os.path.isdir(weights_dir):
        os.makedirs(weights_dir)
    return tensorboard_dir, imgs_dir, weights_dir
