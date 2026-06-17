import numpy as np


def model(pars, x, inD):
    """
    Evaluates the model for given inputs. 

    Inputs
    ------
    params   : array. Parameters to be predicted on.

    Outputs
    -------
    results: array. Model corresponding to the given inputs
    """
    # Load params
    params = np.zeros(inD, dtype=float)
    for i in np.arange(inD):
        params[i] = pars[i]

    return params[0]*x**2 + params[1]*x + params[2]



def prior(cube, ndim, nparams, pmin, pmax, pstep):
    # Cube begins as [0,1] interval -- scale to [pmin, pmax]
    for i in range(ndim):
        cube[i] = cube[i]                                 \
                  * (pmax[pstep>0][i] - pmin[pstep>0][i]) \
                  +  pmin[pstep>0][i]
    return cube


def loglikelihood(cube, ndim, nparams, data, uncert, model):
    ymodel = model(cube)
    loglike = (-0.5 * ((ymodel - data) / uncert)**2).sum()
    return loglike


