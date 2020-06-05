import sys
sys.path.append('/home/gridsan/ssamsi/gaia-dev/sevir-ml/sevir-ml/eie-sevir')
import os
import csv
import glob
import tqdm
import argparse
import numpy as np
import tensorflow as tf
from readers.nowcast_reader import read_data_legacy as read_data
from models.nowcast_unet import create_model
from models.nowcast_gan import generator
from cbhelpers.custom_callbacks import writetofile
from utils.plotutils import set_matplotlib_numpy_prefs,removeaxes
from metrics import probability_of_detection,success_rate,critical_success_index,SSIM,MSE,MAE
from matplotlib import pylab as plt
from sevir.display import get_cmap

def metrics_fns(nout=1):
    def pod74(y_true,y_pred):
        return probability_of_detection(y_true,y_pred,np.array(nout*[74.0],dtype=np.float32))
    def pod133(y_true,y_pred):
        return probability_of_detection(y_true,y_pred,np.array(nout*[133.0],dtype=np.float32))
    def sucr74(y_true,y_pred):
        return success_rate(y_true,y_pred,np.array(nout*[74],dtype=np.float32))
    def sucr133(y_true,y_pred):
        return success_rate(y_true,y_pred,np.array(nout*[133],dtype=np.float32))
    def csi74(y_true,y_pred):
        return critical_success_index(y_true,y_pred,np.array(nout*[74],dtype=np.float32))
    def csi133(y_true,y_pred):
        return critical_success_index(y_true,y_pred,np.array(nout*[133],dtype=np.float32))
    def ssim(y_true,y_pred):
        return SSIM(y_true,y_pred,n_out=nout)
    def mse(y_true,y_pred):
        return MSE(y_true,y_pred)
    def mae(y_true,y_pred):
        return MAE(y_true,y_pred)
    return [pod74,pod133,sucr74,sucr133,csi74,csi133,ssim,mse,mae]

def get_metrics(y_true, y_pred, nout=12):
    fn_list = metrics_fns(nout=nout)
    m = []
    for ii in range(y_true.shape[-1]):
        m.append([METRIC(y_true[...,ii], y_pred[...,ii]).numpy() for METRIC in fn_list])
    
    return m
