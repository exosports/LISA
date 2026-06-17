import numpy as np


def model(params, x):
    """
    Evaluates the model for given inputs. 

    Inputs
    ------
    params   : array. Parameters to be predicted on.

    Outputs
    -------
    results: array. Model corresponding to the given inputs
    """
    return params[0]*x**2 + params[1]*x + params[2]


def loglikelihood(cube, data, uncert, model):
    ymodel = model(cube)
    loglike = (-0.5 * ((ymodel - data) / uncert)**2).sum()
    return loglike


