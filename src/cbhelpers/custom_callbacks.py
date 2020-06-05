import sys
import os
import io
import logging
import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use('agg')
import matplotlib.pylab as plt
from readers.nowcast_reader import read_data_legacy
from utils.plotutils import removeaxes,add_colorbar

sys.path.append('/home/gridsan/groups/EarthIntelligence/datasets/SEVIR')
from sevir.display import get_cmap
from matplotlib.colors import ListedColormap

vil_cmap, vil_norm, vmax, vmin = get_cmap('vil')

def make_callback_dirs(tensorboard_dir, imgs_dir, weights_dir):
    if not os.path.isdir(tensorboard_dir):
        os.makedirs(tensorboard_dir)

    if not os.path.isdir(imgs_dir):
        os.makedirs(imgs_dir)

    if not os.path.isdir(weights_dir):
        os.makedirs(weights_dir)
    return tensorboard_dir, imgs_dir, weights_dir

# this function is just for visualizing a specific dataset during training
def get_viz_inputs(test_file='/home/gridsan/groups/EarthIntelligence/engine/sid/data/test_data.h5'):
    IN, OUT = read_data_legacy(test_file, indices=(4,30), MEAN=33.44, SCALE=47.54)
    return IN,OUT

def save_predictions(epoch, savedir, threshold, model, IN, OUT, MEAN=33.44,SCALE=47.54): #MEAN=33.44, SCALE=47.54):
    logging.info(f'epoch {epoch} : running inference')
    y = model.predict(IN)
    
    # "y" is a list when the unet produces multiple outputs
    if type(y) == list:
        y = np.moveaxis(np.asarray(y), 0, -1)
    y = y*SCALE+MEAN
    logging.info(f'output shape : {y.shape}')
    if len(y.shape) > 4:
        # in replicated output case, the output can be Nx384x384x12x2
        # we only want one of these outputs since they are identical
        y = np.squeeze(y[:,:,:,:,0])
    logging.info(f'output shape : {y.shape}')
    
    # output can have different time steps 12, 6, 3, etc.
    # trim all the arrays we are plotting to the same number of time steps
    OUT = OUT[:,:,:,:y.shape[-1]]
    OUT = OUT*SCALE + MEAN
    logging.info(f'OUT array shape : {OUT.shape}')
    filename = os.path.join(savedir, f'epoch{epoch}.png')
    writetofile(y, OUT, filename=filename) # vil colormap
    return


# Assumes that A=truth, B=predicted
def get_mask(A,B,th=74.):
    mask = A*0
    mask[ (A<th) & (B<th) ] = 0
    mask[ (A>th) & (B>th) ] = 1
    mask[ (A>th) & (B<th) ] = 2
    mask[ (A<th) & (B>th) ] = 3
    return mask

#0 : both fields 0
#1 : both fields 1
#2 : truth=1, pred=0
#3 : truth=0, pred=1
def writetofile(y, OUT, threshold=74., filename='result.png', figsize=(18,9), vmin=0, vmax=255):
    my_cmap = ListedColormap(np.asarray(((1,1,1),
                                             (0.19215686274509805,
                                                  0.5098039215686274,
                                             0.7411764705882353), 
                                             (0.6196078431372549,
                                                  0.792156862745098,
                                             0.8823529411764706),
                                             (0.9019607843137255,
                                                  0.3333333333333333,
                                             0.050980392156862744))))
    nr = 6
    nc = 12
    offset = 0
    hf = plt.figure(figsize=figsize)
    hf.set_tight_layout(True)
    for image_num in range(2):
        truth = np.squeeze(OUT[image_num, ::])
        pred = np.squeeze(y[image_num, ::])

        for ii in range(nc):
            plt.subplot(nr,nc,ii+1+offset)
            plt.imshow(truth[:,:,ii], cmap=vil_cmap, vmin=vmin, vmax=vmax)
            if ii==0:
                removeaxes(ylabel='Truth')
            else:
                removeaxes()

        offset += nc
        for ii in range(nc):
            plt.subplot(nr,nc,ii+1+offset)
            plt.imshow(pred[:,:,ii], cmap=vil_cmap, vmin=vmin, vmax=vmax)
            if ii==0:
                removeaxes(ylabel='Predicted')
            else:
                removeaxes()

        offset += nc
        for ii in range(nc):
            plt.subplot(nr,nc,ii+1+offset)
            plt.imshow(get_mask(truth[:,:,ii], pred[:,:,ii], threshold), cmap=my_cmap)
            plt.xticks([])
            plt.yticks([])
        offset += nc

    plt.savefig(filename, bbox_inches='tight')
    plt.close()
    return

