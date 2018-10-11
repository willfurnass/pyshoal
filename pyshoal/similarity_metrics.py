import numpy as np
import scipy.stats as sp_stats

def check_comparable(vec1, vec2):
    if vec1.shape != vec2.shape and vec1.ndim != 1 and vec2.ndim != 1:
        raise Exception("Arguments must be 1D numpy.ndarrays of the same shape")

def nash_sutcliffe(obs_v, pred_v):
    """
    Nash Sutcliffe Efficiency Index correlation metric

    Nash, J. E. and Sutcliffe, J. V. (1970).  River flow forecasting through conceptual models 
    part I - A discussion of principles, J. Hydrology, 10(3), pp. 282-290, ISSN 0022-1694, 
    DOI 10.1016/0022-1694(70)90255-6
    """
    check_comparable(obs_v, pred_v)
    pred_v_mean = pred_v.mean()
    return 1. - (np.sum(np.power(obs_v - pred_v, 2.)) / np.sum(np.power(obs_v - pred_v_mean, 2.)))

def r_squared(vec1, vec2):
    """Aka coefficient of determination (?).
    
    Is square of Pearson product-moment correlation coefficient.
    """
    check_comparable(vec1, vec2)
    return (sp_stats.pearsonr(vec1, vec2)[0]) ** 2.

def rmsd(vec1, vec2):
    """Root Mean Square Deviation correlation metric

    See:
    Anderson, M. P. and Woessner, W. W. (1992).  Applied Groundwater Modeling: Simulation 
    of Flow and Advective Transport, Academic Press, London, UK.
    """
    check_comparable(vec1, vec2)
    return np.sqrt(np.sum(np.power(vec1 - vec2, 2.)) / vec1.size)

def avg_pc_err(obs_v, pred_v):
    """Average Percent Error correlation metric.
    
    See:
    Kashefipour, S. M. and Falconer, R. A. (2002). Longitudinal dispersion coefficients in 
    natural channels, Water Research, 36(6), pp 1596-1608, ISSN 0043-1354, 
    DOI: 10.1016/S0043-1354(01)00351-7
    """
    check_comparable(obs_v, obs_v)
    return 100 * np.sum(np.abs(obs_v - pred_v)) / np.sum(obs_v)

def integ_sq_errors(vec1, vec2):
    """Integral of squared errors."""
    check_comparable(vec1, vec2)
    return np.sum(np.power(vec1 - vec2, 2.0))
