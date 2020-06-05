"""
Compute 2D histograms to compare prediction and truth
"""
import numpy as np

def compute_stats(hits,
                  misses,
                  false_alarms,
                  correct_rejections,
                  partial_hits=0,
                  partial_hit_weight=1.0,
                  partial_misses=0,
                  partial_miss_weight=0.0
                  ):
    """
    Computes scoring statistics based on hits (H), misses (M),
    false alarms (F) and correct rejections (C).
    
    Optionally, partial hits and misses can also be passes, along with
    weights for each.
    
    In the case of binary scoring (no partial hits/misses)
    
    stats = {'n_truth':H+M,
             'n_pred':H+F,
             'hits':H,
             'misses':M,
             'false_alarms':F,
             'correct_rejections':C,
             'pod':H/(H+M),
             'far':F/(F+H),
             'csi':H/(H+M+F),
             'bias':(H+F)/(H+M)}
    
    If partial hits/misses are included, then  
      
        hits  :  hits   + partial_hits * partial_hit_weight
        misses  :  misses + partial_misses * partial_miss_weight
    
    
    Parameters
    ----------
    hits   scalar
       Number of hits in scene   
    misses   scalar
       Number of misses in scene
    false_alarms   scalar
       Number of false alarms in scene
    correct_rejections   scalar
       Number of correct rejections in scene
    partial_hits scalar
       Number of partial hits
    partial_hits_weight scalar
       Weight of partial hits in pod/far/csi calculation
    partial_misses
       Number of partial misses
    partial_missses_weight scalar
       Weight of partial misses in pod/far/csi calculation
       
    Returns
    -------
    scores  dict
       Dictionary containing statistics                    
    """
    H=hits+partial_hits*partial_hit_weight
    M=misses+partial_misses*partial_miss_weight
    F=false_alarms
    C=correct_rejections
    
    n_truth=1.0*(H+M)
    n_pred=1.0*(H+F)
    n_any=1.0*(H+M+F)
    
    pod = 1.0*H/n_truth if n_truth>0 else 1.0
    far = 1.0*F/n_pred if n_pred>0 else 0.0
    csi = 1.0*H/n_any if n_any>0 else 1.0
    bias = n_pred/n_truth if n_truth>0 else 1.0
    
    return {'n_truth':  n_truth,
            'n_pred':  n_pred,
            'hits':    H,
            'misses':  M,
            'false_alarms':F,
            'correct_rejections':C,
            'pod':pod,
            'far':far,
            'csi':csi,
            'bias':bias}


def compute_histogram(truth,pred,bins=255,**kwargs):
    """
    Compares two np.array's of similar dimensions by computing a 2D histogram
    over pixel values. This function is mainly a wrapper of numpy.histogram2d.
    
    The output is a matrix of counts.  The rows correspond to values (or bins)
    in the turth, and the columns correspond to values (or 
    bins) in the prediction.  The value at pixel i,j in the output 
    matrix is the count of how many times a turth value falls in bin i 
    and a predicted value falls in bin j at co-located pixels.
    
    A "perfect" prediction would yield a hitograms with non-zero counts along 
    the diagonal and zero everywhere else. 
    
    Standard forecast statistics can be computed quickly form the histogram
    for multiple thresholds by passing output to score_histogram
    
    Parameters
    ----------
    truth np.array 
       Input array representing truth.  
    pred np.array 
       Input array representing prediction
    bins  (see numpy.histogram2d)
       The bin specification.  Pasted form numpy docs: 
            If int, the number of bins for the two dimensions (nx=ny=bins).
            If array_like, the bin edges for the two dimensions (x_edges=y_edges=bins).
            If [int, int], the number of bins in each dimension (nx, ny = bins).
            If [array, array], the bin edges in each dimension (x_edges, y_edges = bins).
            A combination [int, array] or [array, int], where int is the number of bins and array is the bin edges.
    kwargs  
       Additional arguments passed to numpy.histogram2d.
    
    Returns
    -------
    H    numpy array nx x ny
       Histogram
    rowedges   numpy array 1 x nx+1
       Array of edges along rows
    coledges   numpy array  1 x ny+1
       Array of edges along columns
    
    """
    if pred.shape!=truth.shape:
        raise ValueError('Inputs must have same dimension %s!=%s'%(pred.shape,truth.shape))
    pred = pred.flatten()
    truth = truth.flatten()
    H,rowedges,coledges=np.histogram2d(truth,pred,bins=bins,**kwargs)
    return H,rowedges,coledges
    

def score_histogram(hist, truth_bins, pred_bins, thresholds):
    """
    Computes scoring statistics for multiple thresholds based on a histogram
    (computing using compute_histogram).  
    
    It is assumed that the rows of the histogram correspond to values in the 
    prediction, and columns correspond to values in the truth.
    
    Parameters
    ----------
    hist  numpy array  nx x ny
        Score histogram (from compute_histogram)
    truth_bins  1 x nx+1        
        Values corresponding to the rows of the histogram
    pred_bins  1 x ny+1
        Values corresponding to the columns of the histogram
    thresholds   array or dict
        If array, scoring thresholds used to compute statistics
        If dict,  {label : threshold}.  labels are used as keys in output
    
    Returns
    -------
    scores   dict
        Dictionary of scores for each threshold
    """
    if type(thresholds)!=dict:
        # Make it a dict
        thresholds = {t:t for t in thresholds}
    scores={}
    for label,thres in thresholds.items():
        thres_row=np.argmax(truth_bins>=thres)# argmax will give index of first 1
        thres_col=np.argmax(pred_bins>=thres) # argmax will give index of first 1
        H=np.sum(hist[thres_row:,thres_col:])
        F=np.sum(hist[:thres_row,thres_col:])
        M=np.sum(hist[thres_row:,:thres_col])
        C=np.sum(hist[:thres_row,:thres_col])
        scores[label]={'threshold':thres}
        scores[label].update(compute_stats(H,M,F,C)) 
    return scores


