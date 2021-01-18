"""
Wrapper for dynesty dynamic nested sampling algorithm of Speagle (2019).

driver(): executes the inference
"""

import sys, os
import numpy as np
import matplotlib.pyplot as plt
import dynesty


def driver(params):
    """
    Dynesty algorithm of Speagle (2019)

    Inputs
    ------
    params: dict. Dictionary of input parameters.  Must include an entry for 
                  bound, kll, loglike, max_iters, min_ess, model, nlive_batch, 
                  nlive_init, pstep, prior, and sample.
                  See user manual for descriptions.

    Outputs
    -------
    outp : array. Each set of accepted parameter values.
    bestp: array. Best parameter values.
    """
    # Setup the sampler
    ndim    = np.sum(params["pstep"] > 0)
    sampler = dynesty.DynamicNestedSampler(params["loglike"], 
                                           params["prior"], 
                                           ndim, 
                                           bound=params["bound"], 
                                           sample=params["sample"])

    # Run it
    sampler.run_nested(nlive_init=params["nlive_init"], 
                       nlive_batch=params["nlive_batch"], 
                       maxiter=params["max_iters"], 
                       n_effective=params["min_ess"])
    results = sampler.results

    # Posterior and best parameters
    bestp = results["samples"][np.argmax(results["logl"])]
    samps = results["samples"]
    nsamp = samps.shape[0]
    # From cornerplot in dynesty/plotting.py 
    try:
        weights = np.exp(results['logwt'] - results['logz'][-1])
    except:
        weights = results['weights']
    # Remove cases w/ 0 weight
    samps   = samps  [weights!=0]
    weights = weights[weights!=0]

    # As in MultiNest, resample to equal weights by considering nsamples*int(w) 
    # repetitions
    outp = []
    for i in range(len(weights)):
        outp.extend([samps[i] for j in range(int(nsamp * weights[i]))])
    outp = np.asarray(outp)

    if params["kll"] is not None:
        for i in range(outp.shape[0]):
            params["kll"].update(params["model"](outp[i], fullout=True))
    outp = outp.T

    return outp, bestp


