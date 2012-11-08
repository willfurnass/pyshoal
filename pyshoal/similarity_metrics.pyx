import numpy as np

def nash_sutcliffe(obs_v, model_v):
    if obs_v.shape != model_v.shape:
        raise Exception("numpy.ndarray arguments must have same shape")
    
    model_v_mean = model_v.mean()
    return 1. - (np.sum(np.power(obs_v - model_v, 2.)) / np.sum(np.power(obs_v - model_v_mean, 2.)))
