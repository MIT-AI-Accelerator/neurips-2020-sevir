import os
import h5py
import numpy as np

def get_data(train_data, test_data, num_train=1024, pct_validation=0.2, dtype=np.float32):
    # read data: this function returns scaled data
    # what about shuffling ? 
    logging.info(f'rank {global_rank} reading images')
    t0 = time.time()
    train_IN, train_OUT = read_data(train_data, end=num_train, dtype=dtype) 
    t1 = time.time()
    logging.info(f'read time : {t1-t0}')

    # Make the validation dataset the last pct_validation of the training data
    val_idx = int((1-pct_validation)*train_IN.shape[0])
    
    val_IN = train_IN[val_idx:, ::]
    train_IN = train_IN[:val_idx, ::]

    val_OUT = train_OUT[val_idx:, ::]
    train_OUT = train_OUT[:val_idx, ::]

    return (train_IN,train_OUT,val_IN,val_OUT)


def read_data(filename, rank=0, size=1, end=None, dtype=np.float32, MEAN=33.44, SCALE=47.54):
    x_keys = ['IN']
    y_keys = ['OUT']
    s = np.s_[rank:end:size]
    with h5py.File(filename, mode='r') as hf:
        IN  = hf['IN'][s]
        OUT = hf['OUT'][s]
    IN = (IN.astype(dtype)-MEAN)/SCALE
    OUT = (OUT.astype(dtype)-MEAN)/SCALE
    return IN,OUT

