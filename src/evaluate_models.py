global_rank = 0
global_size = 1

import sys
sys.path.append('/home/gridsan/groups/EarthIntelligence/datasets/SEVIR/')
import os
import time
import tables
import logging
import argparse
import numpy as np
import tensorflow as tf
import pandas as pd
from tqdm import tqdm
from losses import lpips
from metrics import probability_of_detection,success_rate
from metrics.lpips_metric import get_lpips
from metrics.histogram import compute_histogram,score_histogram
from utils.utils import setuplogging,log_args

norm = {'scale':47.54,'shift':33.44}
val_file = '/home/gridsan/groups/EarthIntelligence/engine/markv/ML-SEVIR/data/interim/nowcast_testing.h5'
    
def ssim(y_true,y_pred,maxVal,**kwargs):
    yt=tf.convert_to_tensor(y_true.astype(np.uint8))
    yp=tf.convert_to_tensor(y_pred.astype(np.uint8))
    #s = tf.image.ssim( yt, yp, max_val=maxVal[0], **kwargs)
    s=tf.image.ssim_multiscale(
              yt, yp, max_val=maxVal[0], filter_size=11, filter_sigma=1.5, k1=0.01, k2=0.03
    )
    return tf.reduce_mean(s)

def MAE(y_true,y_pred,dum):
    return tf.reduce_mean(tf.keras.losses.MAE(y_true,y_pred))


def MSE(y_true,y_pred,dum):
    return tf.reduce_mean(tf.keras.losses.MSE(y_true,y_pred))


def run_metric( metric, thres, y_true, y_pred, batch_size):
    result = 0.0
    Ltot = 0.0
    n_batches = int(np.ceil(y_true.shape[0]/batch_size))
    print('Running metric ',metric.__name__,'with thres=',thres)
    for b in range(n_batches):
        start = b*batch_size
        end   = min((b+1)*batch_size,y_true.shape[0])
        L = end-start
        yt = y_true[start:end]
        yp = y_pred[start:end]
        result += metric(yt.astype(np.float32),yp,np.array([thres],dtype=np.float32))*L
        Ltot+=L
    return (result / Ltot).numpy() 

def run_histogram(y_true, y_pred, batch_size=1000,bins=range(255)):
    L = len(bins)-1
    H = np.zeros( (L,L),dtype=np.float64) 
    n_batches = int(np.ceil(y_true.shape[0]/batch_size))
    print('Computing histogram ')
    for b in range(n_batches):
        start = b*batch_size
        end   = min((b+1)*batch_size,y_true.shape[0])
        yt = y_true[start:end]
        yp = y_pred[start:end]
        Hi,rb,cb = compute_histogram(yt,yp,bins)
        H+=Hi
    return H,rb,cb


def get_input(filename, arr_name=None, num_read=None):
    # this returns un-scaled data
    with tables.open_file(filename, mode='r') as reader:
        if num_read is not None:
            IN = reader.root[arr_name].read(start=0, stop=num_read)
        else:
            IN = reader.root[arr_name].read()
    return IN

def get_model(weights):
    return tf.keras.models.load_model(weights,compile=False,custom_objects={"tf": tf})

def main(args):
    logging.info('get data')
    IN = get_input(args.val_data, arr_name='IN', num_read=args.num_read)

    logging.info(f'IN : {IN.shape}')
    logging.info('scaling and shifting')
    t0 = time.time()
    IN = (IN.astype(np.float32)-norm['shift'])/norm['scale']
    t1 = time.time()
    logging.info(f'scaling and shifting time : {t1-t0}')
    
    logging.info('predict')
    if args.loss=='pers':
        logging.info('no need to load any model : persistence loss')
        # only keep the data for persistence
        IN = IN[...,11:12] 
        IN = IN*norm['scale']+norm['shift']
        y_pred = np.concatenate(12*[IN], axis=-1) # this may cause an OOM explosion
        IN = None # just to release memory ...
        logging.info(f'persistence data : {y_pred.shape}')
    else:
        logging.info('get model')
        model = get_model(args.model)
        t0 = time.time()
        y_pred = model.predict(IN, batch_size=32, verbose=2)
        # scale predictions back to original scale and mean
        if type(y_pred) == list:
            # this happens when loss is mse+style
            y_pred = y_pred[0]
        y_pred = y_pred*norm['scale']+norm['shift']
        t1 = time.time()
        logging.info(f'predict time : {t1-t0}')
        logging.info(f'y_pred : {y_pred.shape}')
        model = None # cleanup for random reason
    
    logging.info('get truth data')
    t0 = time.time()
    y_test = get_input(args.val_data, arr_name='OUT', num_read=args.num_read)
    t1 = time.time()
    logging.info(f'time elapsed : {t1-t0}')    
    logging.info(f'y_test : {y_test.shape}')

    logging.info('get metrics')
    # calculate metrics in batches
    
    test_scores_lead = {}
    t0 = time.time()
    # Loop over 12 lead times
    model = lpips.PerceptualLoss(model='net-lin', net='alex', use_gpu=False)#True, gpu_ids=[1])
    for lead in range(12):
        test_scores={}
    
        yt = y_test[...,lead:lead+1] # truth data
        yt = yt.astype(np.float32) # this is in the original range. no need to rescale
        yp = y_pred[...,lead:lead+1] # predictions have been scaled earlier
        
        test_scores['ssim'] = run_metric(ssim, [255], yt,yp, batch_size=32)
        test_scores['mse'] = run_metric(MSE, 255, yt, yp, batch_size=32)
        test_scores['mae'] = run_metric(MAE, 255, yt, yp, batch_size=32)
        test_scores['lpips'] = get_lpips(model,yp,yt,batch_size=32,n_out=1)[0] # becuase this is scalar
        
        H,rb,cb=run_histogram(yt,yp,bins=range(255))
        thresholds = [16,74,133,160,181,219]
        scores = score_histogram(H,rb,cb,thresholds)
        for t in thresholds:
            test_scores['pod%d' % t] = scores[t]['pod']
            test_scores['sucr%d' % t] = 1-scores[t]['far']
            test_scores['csi%d' % t] = scores[t]['csi']
            test_scores['bias%d' % t] = scores[t]['bias']
        
        test_scores_lead[lead]=test_scores

    t1 = time.time()
    logging.info(f'metrics time : {t1-t0}')
    
    metrics_file = os.path.join(args.log_dir, f'test_scores_{args.loss}.csv')
    logging.info(f'saving to : {metrics_file}')
    df = pd.DataFrame({k:[v] for k,v in test_scores_lead.items()})
    df.to_csv(metrics_file)
    
    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--loss', type=str, help='name of loss function used',
                            choices=['mse', 'style', 'mse+style', 'gan', 'pers'])
    parser.add_argument('--val-data', type=str, help='path to val data')
    parser.add_argument('--log-dir', type=str, help='path to log dir created for this job')
    parser.add_argument('--num-read', type=int, default=None)
    parser.add_argument('--basedir', type=str, default='/home/gridsan/ssamsi/projects/ML-SEVIR/src/logs')
    args = parser.parse_args()

    setuplogging(os.path.join(args.log_dir, f'{global_rank}.log'))
    log_args(args)

    #model_dict = {'mse' :'507999-mse/weights/weights_100.h5',
    #                  'style': '508001-vgg/weights/weights_60.h5',
    #                  'mse+style': '508000-mse_vgg/weights/weights_100.h5',
    #                  'gan': '508057-gan-DIST/weights/trained_generator-100.h5',
    #                  'pers': 'persistence_model'
    #    }

    model_dict = {'mse' :'516796-mse/weights/weights_100.h5',
                      'style': '516797-vgg/weights/weights_100.h5',
                      'mse+style': '516799-mse_vgg/weights/weights_100.h5',
                      'gan': '516931-gan-DIST/weights/trained_generator-299.h5',
                      'pers': 'persistence_model'
        }

    if args.loss is not 'pers':
        args.model = os.path.join(args.basedir, model_dict[args.loss])
    else:
        args.model = None
    main(args)
    
