import numpy as np
import scipy.stats as sp_stats

from __future__ import division

def nash_sutcliffe(obs_v, pred_v):
    """
    Nash Sutcliffe Efficiency Index correlation metric

    Nash, J. E. and Sutcliffe, J. V. (1970).  River flow forecasting through conceptual models 
    part I — A discussion of principles, J. Hydrology, 10(3), pp. 282-290, ISSN 0022-1694, 
    DOI 10.1016/0022-1694(70)90255-6
    """
    if obs_v.shape != pred_v.shape and obs_v.ndim != 1 and pred_v.ndim != 1:
        raise Exception("Arguments must be 1D numpy.ndarrays of the same shape")
    
    pred_v_mean = pred_v.mean()
    return 1. - (np.sum(np.power(obs_v - pred_v, 2.)) / np.sum(np.power(obs_v - pred_v_mean, 2.)))

def r_squared(a, b):
    """Aka coefficient of determination (?).
    
    Is square of Pearson product-moment correlation coefficient.
    """
    if a.shape != b.shape and a.ndim != 1 and b.ndim != 1:
        raise Exception("Arguments must be 1D numpy.ndarrays of the same shape")
    return (sp_stats.pearsonr(a, b)[0]) ** 2.

def rmsd(a, b):
    """Root Mean Square Deviation correlation metric

    See:
    Anderson, M. P. and Woessner, W. W. (1992).  Applied Groundwater Modeling: Simulation 
    of Flow and Advective Transport, Academic Press, London, UK.
    """
    if a.shape != b.shape and a.ndim != 1 and b.ndim != 1:
        raise Exception("Arguments must be 1D numpy.ndarrays of the same shape")
    return np.sqrt(np.sum(np.power(a - b, 2.)) / a.size)

def avg_pc_err(obs_v, pred_v):
    """
    Average Percent Error correlation metric.
    
    See:
    Kashefipour, S. M. and Falconer, R. A. (2002). Longitudinal dispersion coefficients in 
    natural channels, Water Research, 36(6), pp 1596-1608, ISSN 0043-1354, 
    DOI: 10.1016/S0043-1354(01)00351-7
    """
    if obs_v.shape != pred_v.shape and obs_v.ndim != 1 and pred_v.ndim != 1:
        raise Exception("Arguments must be 1D numpy.ndarrays of the same shape")
    return 100 * np.sum(np.abs(obs_v - pred_v)) / np.sum(obs_v)
