"""
Generator for nowcast dataset
"""
import sys
import numpy as np
import tensorflow as tf
import datetime
from sevir.utils import SEVIRSequence

class NowcastGenerator(SEVIRSequence):
    """
    Generator that loads full VIL sequences, and spilts each
    event into three training samples, each 12 frames long.

    Event Frames:  [-----------------------------------------------]
                   [----13-----][---12----]
                               [----13----][----12----]
                                          [-----13----][----12----]
    """
    def __getitem__(self, idx):
        """

        """
        X,_ = super(NowcastGenerator, self).__getitem__(idx)  # N,L,W,49
        x1,x2,x3 = X[0][:,:,:,:13],X[0][:,:,:,12:25],X[0][:,:,:,24:37]
        y1,y2,y3 = X[0][:,:,:,13:25],X[0][:,:,:,25:37],X[0][:,:,:,37:49]
        Xnew = np.concatenate((x1,x2,x3),axis=0)
        Ynew = np.concatenate((y1,y2,y3),axis=0)
        return [Xnew],[Ynew]

def get_nowcast_train_generator(sevir_catalog,
                                sevir_location,
                                batch_size=8,
                                start_date=None,
                                end_date=datetime.datetime(2019,6,1) ):
    filt = lambda c:  c.pct_missing==0 # remove samples with missing radar data
    return NowcastGenerator(catalog=sevir_catalog,
                            sevir_data_home=sevir_location,
                            x_img_types=['vil'],
                            y_img_types=['vil'],
                            batch_size=batch_size,
                            start_date=start_date,
                            end_date=end_date,
                            catalog_filter=filt)

def get_nowcast_test_generator(sevir_catalog,
                               sevir_location,
                               batch_size=8,
                               start_date=datetime.datetime(2019,6,1),
                               end_date=None):
    filt = lambda c:  c.pct_missing==0 # remove samples with missing radar data
    return NowcastGenerator(catalog=sevir_catalog,
                            sevir_data_home=sevir_location,
                            x_img_types=['vil'],
                            y_img_types=['vil'],
                            batch_size=batch_size,
                            start_date=start_date,
                            end_date=end_date,
                            catalog_filter=filt)



