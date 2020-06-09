"""
data reader for synrad using SEVIR
"""

import logging
import os
os.environ["HDF5_USE_FILE_LOCKING"]='FALSE'
import h5py
import numpy as np


def get_data(train_data, pct_validation=0.2, rank=0, size=1, end=None ):
    logging.info('Reading datasets')
    train_IN, train_OUT = read_data(train_data, rank=rank, size=size, end=end)

    # Make the validation dataset the last pct_validation of the training data
    val_idx = int((1-pct_validation)*len(train_OUT['vil']))
    val_IN={}
    val_OUT={}
    for k in train_IN:
        train_IN[k],val_IN[k]=train_IN[k][:val_idx],train_IN[k][val_idx:]
    for k in train_OUT:
        train_OUT[k],val_OUT[k]=train_OUT[k][:val_idx],train_OUT[k][val_idx:]
    
    logging.info('data loading completed')
    return (train_IN,train_OUT,val_IN,val_OUT)


def read_data(filename, rank=0, size=1, end=None,dtype=np.float32):
    x_keys = ['ir069','ir107','lght']
    y_keys = ['vil']
    s = np.s_[rank:end:size]
    with h5py.File(filename, 'r') as hf:
        IN  = {k:hf[k][s].astype(np.float32) for k in x_keys}
        OUT = {k:hf[k][s].astype(np.float32) for k in y_keys}
    return IN,OUT

    
    


