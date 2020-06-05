"""
Generators for synrad datasets
"""

import argparse
import logging

import os
import h5py
os.environ["HDF5_USE_FILE_LOCKING"]='FALSE'

import sys
import datetime
import numpy as np
import tensorflow as tf
from sevir.utils import SEVIRSequence

def get_synrad_train_generator(sevir_catalog,
                                sevir_location,
                                batch_size=32,
                                start_date=None,
                                end_date=datetime.datetime(2019,6,1) ):
    filt = lambda c:  c.pct_missing==0 # remove samples with missing data
    return SEVIRSequence(catalog=sevir_catalog,
                         sevir_data_home=sevir_location,
                         x_img_types=['ir069','ir107','lght'],
                         y_img_types=['vil'],
                         batch_size=batch_size,
                         start_date=start_date,
                         end_date=end_date,
                         catalog_filter=filt,
                         unwrap_time=True)

def get_synrad_test_generator(sevir_catalog,
                               sevir_location,
                               batch_size=32,
                               start_date=datetime.datetime(2019,6,1),
                               end_date=None):
    filt = lambda c:  c.pct_missing==0 # remove samples with missing radar data
    return SEVIRSequence(catalog=sevir_catalog,
                         sevir_data_home=sevir_location,
                         x_img_types=['ir069','ir107','lght'],
                         y_img_types=['vil'],
                         batch_size=batch_size,
                         start_date=start_date,
                         end_date=end_date,
                         catalog_filter=filt,
                         unwrap_time=True)
