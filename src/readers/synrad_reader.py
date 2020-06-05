"""
data reader for synrad using SEVIR
"""

import os
os.environ["HDF5_USE_FILE_LOCKING"]='FALSE'
import h5py
import numpy as np



def read_data(filename, rank, size, end=None,dtype=np.float32):
    x_keys = ['ir069','ir107','lght']
    y_keys = ['vil']
    s = np.s_[rank:end:size]
    with h5py.File(filename, 'r') as hf:
        IN  = {k:hf[k][s].astype(np.float32) for k in x_keys}
        OUT = {k:hf[k][s].astype(np.float32) for k in y_keys}
    return IN,OUT

    
    


