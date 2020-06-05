import os
import h5py
import numpy as np
import tensorflow as tf
from tensorflow import data

def read_data_0(filename, rank=0, size=1, batch_size=4, dtype=np.float32,
                  MEAN=33.44, SCALE=47.54, scale=True):
    
    if not os.path.isfile(filename):
        return None,None
    
    with h5py.File(filename, mode='r') as reader:
        in_arr = reader.get('IN')
        out_arr = reader.get('OUT')
        
        my_fraction = in_arr.shape[0]//size
        start = my_fraction*rank
        stop = start + my_fraction

        # Read data and cast to appropriate type
        # Issue 1 : This can cause OOM on CPU if we read in too many records. How do we handle this ?
        # Issue 2 : Do not use tf.cast() because this will push data to the GPU and cause OOM with
        #           even smaller numbers of records
        x = in_arr[start:stop,::].astype(dtype)
        y = out_arr[start:stop,::].astype(dtype)
        
        if scale:
            x = (x-MEAN)/SCALE
            y = (y-MEAN)/SCALE

        IN  = data.Dataset.from_tensor_slices(x).shuffle(my_fraction).batch(batch_size)
        OUT = data.Dataset.from_tensor_slices(y).shuffle(my_fraction).batch(batch_size)
    
    return IN,OUT

def read_data(filename, rank=0, size=1, batch_size=4, dtype=np.float32,
                  MEAN=33.44, SCALE=47.54, scale=True):
    
    if not os.path.isfile(filename):
        return None,None
    
    with h5py.File(filename, mode='r') as reader:
        in_arr = reader.get('IN')
        out_arr = reader.get('OUT')
        
        my_fraction = in_arr.shape[0]//size
        start = my_fraction*rank
        stop = start + my_fraction

        # Read data and cast to appropriate type
        # Issue 1 : This can cause OOM on CPU if we read in too many records. How do we handle this ?
        # Issue 2 : Do not use tf.cast() because this will push data to the GPU and cause OOM with
        #           even smaller numbers of records
        IN = in_arr[start:stop,::].astype(dtype)
        OUT = out_arr[start:stop,::].astype(dtype)
        
        if scale:
            IN = (IN-MEAN)/SCALE
            OUT = (OUT-MEAN)/SCALE
    
    return IN,OUT

