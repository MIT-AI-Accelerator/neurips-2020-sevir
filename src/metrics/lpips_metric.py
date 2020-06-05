import os
import sys
import torch
import numpy as np
from losses import lpips

def to_scaled_tensor(imA):
    # convert to RGB, scale images and cast to pytorch Tensor
    # expected shape is Nx3xHxW
    # first convet to fake RGB
    imA = np.moveaxis(np.transpose(np.tile(np.expand_dims(imA,axis=-1), (1,1,1,3))),-1,0)
    imA = 2*(imA - 127.5)/255 # conver to the range -1:1
    return torch.FloatTensor(imA)

def get_dist(model, yt, yp, n_out):
    # this will take each time step, convert it to RGB, calculate the distance
    d = np.zeros(n_out)
    for ii in range(n_out):
        truth = to_scaled_tensor(yt[...,ii])
        pred = to_scaled_tensor(yp[...,ii])
        d[ii] = model.forward(truth, pred).cpu().detach().numpy().mean()
    # returns the average over the batch for each time step
    return d

# yp : prediction from tensorflow model (assuming these are scaled already)
def get_lpips(model, yp,yt,batch_size=32,n_out=12):    
    d = np.zeros(n_out, dtype=np.float32)

    #model = lpips.PerceptualLoss(model='net-lin', net='alex', use_gpu=True, gpu_ids=[0])

    # do this in batches and average over all images
    # inputs should Nx3xHxW
    num_batches = 0
    for ii in range(0, yt.shape[0], batch_size):
        start = ii
        stop = np.min([start+batch_size, yt.shape[0]])
        d += get_dist(model, yp[start:stop,...], yt[start:stop,...], n_out)
        num_batches += 1
    d = d/num_batches
    return d

