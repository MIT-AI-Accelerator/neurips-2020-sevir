"""
Various metrics for evaluating models trained on SEVIR
"""

import tensorflow as tf

"""
Standard contingincy table-based metrics used in forecast verification
https://www.nws.noaa.gov/oh/rfcdev/docs/Glossary_Verification_Metrics.pdf
"""

def probability_of_detection(y_true,y_pred,threshold):
    """
    
    Inputs:
    -------
    y_true:     [N,L,L,D]
    y_pred:     [N,L,L,D]
    threshold:  [D,]
    Outputs
    -------
    pod=hits/(hits+misses) averaged over the D channels
    """
    return tf.reduce_mean(run_metric_over_channels(y_true,y_pred,threshold,_pod))

def success_rate(y_true,y_pred,threshold):
    """
    a.k.a    1 - (false alarm rate)
    Inputs:
    -------
    y_true:     [N,L,L,D]
    y_pred:     [N,L,L,D]
    threshold:  [D,]
    Outputs
    -------
    sucr=hits/(hits+false_alarms) averaged over the D channels
    """
    return tf.reduce_mean(run_metric_over_channels(y_true,y_pred,threshold,_sucr))

def critical_success_index(y_true,y_pred,threshold):
    """
    
    Inputs:
    -------
    y_true:     [N,L,L,D]
    y_pred:     [N,L,L,D]
    threshold:  [D,]
    Outputs
    -------
    pod=hits/(hits+misses+false_alarms) averaged over the D channels
    """
    return tf.reduce_mean(run_metric_over_channels(y_true,y_pred,threshold,_csi))

def BIAS(y_true,y_pred,threshold):
    """
    Computes the 2^( mean(log BIAS/log 2) )
    
    Inputs:
    -------
    y_true:     [N,L,L,D]
    y_pred:     [N,L,L,D]
    threshold:  [D,]
    Outputs
    -------
    pod=(hits+false_alarms)/(hits+misses) pow(2)-log-averaged over the D channels
    """
    logbias = tf.math.log(run_metric_over_channels(y_true,y_pred,threshold,_bias))/tf.math.log(2.0)
    return tf.math.pow( 2.0, tf.reduce_mean(logbias))

def run_metric_over_channels(y_true,y_pred,threshold,metric):
    """
    
    Inputs:
    -------
    y_true:     [N,L,L,D]
    y_pred:     [N,L,L,D]
    threshold:  [D,]
    Outputs
    -------
    [D,] tensor of metrics computed over each channel
    """
    # permute channels to first dim to work with tf.map_fn
    elems = (tf.transpose(y_true,(3,0,1,2)),
             tf.transpose(y_pred,(3,0,1,2)),
             threshold)
    # Average over channels
    return tf.map_fn(metric,elems,dtype=tf.float32)


def _pod(X):
    """
    Single channel version of probability_of_detection
    Inputs:
    -------
    tuple X = (y_true,y_pred,T) where
                        y_true:     [N,L,L]
                        y_pred:     [N,L,L]
                        T:          [1,]
    """
    y_true,y_pred,T=X
    t,p=_threshold(y_true,y_pred,T)
    hits = tf.reduce_sum(t*p)
    misses = tf.reduce_sum( t*(1-p) )
    return (hits+1e-6)/(hits+misses+1e-6)


def _sucr(X):
    """
    Single channel version of success_rate
    Inputs:
    -------
    tuple X = (y_true,y_pred,T) where
                        y_true:     [N,L,L]
                        y_pred:     [N,L,L]
                        T:          [1,]
    """
    y_true,y_pred,T=X
    t,p=_threshold(y_true,y_pred,T)
    hits = tf.reduce_sum(t*p)
    fas = tf.reduce_sum( (1-t)*p )
    return (hits+1e-6)/(hits+fas+1e-6)

def _csi(X):
    """
    Single channel version of csi
    Inputs:
    -------
    tuple X = (y_true,y_pred,T) where
                        y_true:     [N,L,L]
                        y_pred:     [N,L,L]
                        T:          [1,]
    """
    y_true,y_pred,T=X
    t,p=_threshold(y_true,y_pred,T)
    hits = tf.reduce_sum(t*p)
    misses = tf.reduce_sum( t*(1-p) )
    fas = tf.reduce_sum( (1-t)*p )
    return (hits+1e-6)/(hits+misses+fas+1e-6)

def _bias(X):
    """
    Single channel version of csi
    Inputs:
    -------
    tuple X = (y_true,y_pred,T) where
                        y_true:     [N,L,L]
                        y_pred:     [N,L,L]
                        T:          [1,]
    """
    y_true,y_pred,T=X
    t,p=_threshold(y_true,y_pred,T)
    hits = tf.reduce_sum(t*p)
    misses = tf.reduce_sum( t*(1-p) )
    fas = tf.reduce_sum( (1-t)*p )
    return (hits+fas+1e-6)/(hits+misses+1e-6)

def _threshold(X,Y,T):
    """
    Returns binary tensors t,p the same shape as X & Y.  t = 1 whereever
    X > t.  p =1 wherever Y > t.  p and t are set to 0 whereever EITHER
    t or p are nan.   This is useful for counts that don't involve correct
    rejections.
    """
    t=tf.math.greater_equal(X, T)
    t=tf.dtypes.cast(t, tf.float32)
    p=tf.math.greater_equal(Y, T)
    p=tf.dtypes.cast(p, tf.float32)
    is_nan = tf.math.logical_or(tf.math.is_nan(X),tf.math.is_nan(Y))
    t = tf.where(is_nan,tf.zeros_like(t,dtype=tf.float32),t)
    p = tf.where(is_nan,tf.zeros_like(p,dtype=tf.float32),p)
    return t,p
