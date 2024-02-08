import numpy as np
import dnest4


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


def prior(ndim, pmin, pmax):
    return np.random.uniform(size=ndim) * (pmax - pmin) + pmin


def loglikelihood(cube, data, uncert, model):
    ymodel = model(cube)
    loglike = (-0.5 * ((ymodel - data) / uncert)**2).sum()
    return loglike

def perturb(coords, ndim, width):
    i = np.random.randint(ndim)
    coords[i] += (width[i])*dnest4.randh()
    # Note: use the return value of wrap, unlike in C++
    coords[i] = dnest4.wrap(coords[i], -0.5*width[i], 0.5*width[i])
    return 0.0


