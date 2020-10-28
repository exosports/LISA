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


def prior(cube, pmin, pmax, pstep):
    cube = cube.copy()
    # Cube begins as [0,1] interval -- scale to [pmin, pmax]
    for i in range(cube.shape[0]):
        cube[i] = cube[i]                                 \
                  * (pmax[pstep>0][i] - pmin[pstep>0][i]) \
                  +  pmin[pstep>0][i]
    return cube


def loglikelihood(cube, data, uncert, model):
    ymodel = model(cube)
    loglike = (-0.5 * ((ymodel - data) / uncert)**2).sum()
    return loglike


