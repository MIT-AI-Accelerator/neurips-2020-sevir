import os
import h5py
import numpy as np

def read_data_legacy(filename, start=0, num_read=1, indices=None, dtype=np.float32,
                  MEAN=33.44, SCALE=47.54, scale=True):
    if not os.path.isfile(filename):
        return None,None

    with h5py.File(filename, mode='r') as reader:
        if indices is None:
            indices = np.arange(start, start+num_read)
        else:
            indices = np.asarray(indices)
        in_arr = reader.get('IN')
        IN  = in_arr[indices,::].astype(dtype)
        out_arr = reader.get('OUT')
        OUT  = out_arr[indices,::].astype(dtype)
    if scale:
        IN  = (IN-MEAN)/SCALE
        OUT = (OUT-MEAN)/SCALE
    return IN,OUT

def read_data_slow(filename, rank, size, end=None, dtype=np.float32, MEAN=33.44, SCALE=47.54):
    #MEAN=21.11, SCALE=41.78):
    x_keys = ['IN']
    y_keys = ['OUT']
    s = np.s_[rank:end:size]
    with h5py.File(filename, mode='r') as hf:
        IN  = hf['IN'][s]
        OUT = hf['OUT'][s]
    IN = (IN.astype(dtype)-MEAN)/SCALE
    OUT = (OUT.astype(dtype)-MEAN)/SCALE
    return IN,OUT

def read_data_v1(filename, rank, size, end=None, dtype=np.float32, MEAN=33.44, SCALE=47.54):
    # this might be causing OOM issues
    with h5py.File(filename, mode='r') as hf:
        IN  = hf['IN'][:]
        IN = IN[rank:end:size]
        OUT = hf['OUT'][:]
        OUT = OUT[rank:end:size]
        
    IN = (IN.astype(dtype)-MEAN)/SCALE
    OUT = (OUT.astype(dtype)-MEAN)/SCALE
    return IN,OUT

def read_data_v2(filename, rank, size, end=None, dtype=np.float32, MEAN=33.44, SCALE=47.54):
    with h5py.File(filename, mode='r') as hf:
        IN  = hf['IN'][rank:end:size]
        OUT = hf['OUT'][rank:end:size]
    
    IN = (IN.astype(dtype)-MEAN)/SCALE
    OUT = (OUT.astype(dtype)-MEAN)/SCALE
    return IN,OUT

def read_data(filename, rank, size, end=None, dtype=np.float32, MEAN=33.44, SCALE=47.54):
    with h5py.File(filename, mode='r') as hf:
        n,nr,nc,nz = hf['IN'].shape
        myfraction = n//size
        start = myfraction*rank
        stop = start + myfraction
        IN  = hf['IN'][start:stop,::]
        OUT = hf['OUT'][start:stop,::]
    
    IN = (IN.astype(dtype)-MEAN)/SCALE
    OUT = (OUT.astype(dtype)-MEAN)/SCALE
    return IN,OUT
