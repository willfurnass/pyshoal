import numpy as np
import scipy.stats as sp_stats

def nash_sutcliffe(obs_v, model_v):
    if obs_v.shape != model_v.shape:
        raise Exception("numpy.ndarray arguments must have same shape")
    
    model_v_mean = model_v.mean()
    return 1. - (np.sum(np.power(obs_v - model_v, 2.)) / np.sum(np.power(obs_v - model_v_mean, 2.)))

def r_squared(a, b):
    """Aka coefficient of determination.
    
    Is square of Pearson product-moment correlation coefficient.
    """
    if a.shape != b.shape:
        raise Exception("numpy.ndarray arguments must have same shape")
    return (sp_stats.pearsonr(a,b)[0]) ** 2.

