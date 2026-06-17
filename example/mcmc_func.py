"""
Contains function to be evaluated at each MCMC iteration. 

"""

import sys, os
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
    # Input must be 2D
    if len(params.shape) == 1:
        params = np.expand_dims(params, 0)

    return params[:,0][:, None]*x**2 + params[:,1][:, None]*x + params[:,2][:, None]


