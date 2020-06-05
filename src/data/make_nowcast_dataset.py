"""
Makes training and test dataset for nowcasting model using SEVIR
"""

# -*- coding: utf-8 -*-
import argparse
import logging

import os
import h5py
os.environ["HDF5_USE_FILE_LOCKING"]='FALSE'

import sys
import numpy as np
import tensorflow as tf
from nowcast_generator import get_nowcast_train_generator,get_nowcast_test_generator

parser = argparse.ArgumentParser(description='Make nowcast training & test datasets using SEVIR')
parser.add_argument('--sevir_data', type=str, help='location of SEVIR dataset',default='../../data/sevir')
parser.add_argument('--sevir_catalog', type=str, help='location of SEVIR dataset',default='../../data/CATALOG.csv')
parser.add_argument('--output_location', type=str, help='location of SEVIR dataset',default='../../data/interim')
parser.add_argument('--n_chunks', type=int, help='Number of chucks to use (increase if memory limited)',default=20)


args = parser.parse_args()

def main():
    """ 
    Runs data processing scripts to extract training set from SEVIR
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')
    trn_generator = get_nowcast_train_generator(sevir_catalog=args.sevir_catalog,
                                                sevir_location=args.sevir_data)
    tst_generator = get_nowcast_test_generator(sevir_catalog=args.sevir_catalog,
                                               sevir_location=args.sevir_data)
    
    logger.info('Reading/writing training data to %s' % ('%s/nowcast_training.h5' % args.output_location))
    read_write_chunks('%s/nowcast_training.h5' % args.output_location,trn_generator,args.n_chunks)
    logger.info('Reading/writing testing data to %s' % ('%s/nowcast_testing.h5' % args.output_location))
    read_write_chunks('%s/nowcast_testing.h5' % args.output_location,tst_generator,args.n_chunks)


def read_write_chunks( filename, generator, n_chunks ):
    logger = logging.getLogger(__name__)
    chunksize = len(generator)//n_chunks
    # get first chunk
    logger.info('Gathering chunk 0/%s:' % n_chunks)
    X,Y=generator.load_batches(n_batches=chunksize,offset=0,progress_bar=True)
    # Create datasets
    with h5py.File(filename, 'w') as hf:
      hf.create_dataset('IN', data=X[0],  maxshape=(None,X[0].shape[1],X[0].shape[2],X[0].shape[3]))
      hf.create_dataset('OUT', data=Y[0], maxshape=(None,Y[0].shape[1],Y[0].shape[2],Y[0].shape[3]))
    # Gather other chunks
    for c in range(1,n_chunks+1):
      offset = c*chunksize
      n_batches = min(chunksize,len(generator)-offset)
      if n_batches<0: # all done
        break
      logger.info('Gathering chunk %d/%s:' % (c,n_chunks))
      X,Y=generator.load_batches(n_batches=n_batches,offset=offset,progress_bar=True)
      with h5py.File(filename, 'a') as hf:
            hf['IN'].resize((hf['IN'].shape[0] + X[0].shape[0]), axis = 0)
            hf['OUT'].resize((hf['OUT'].shape[0] + Y[0].shape[0]), axis = 0)
            hf['IN'][-X[0].shape[0]:]  = X[0]
            hf['OUT'][-Y[0].shape[0]:] = Y[0]


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    main()
