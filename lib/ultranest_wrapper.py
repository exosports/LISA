"""
Wrapper for UltraNest algorithm of Buchner (2014, 2016, 2019).

driver(): executes the inference
"""

import sys, os
import numpy as np
import matplotlib.pyplot as plt
import ultranest


def driver(params):
    """
    UltraNest algorithm of Buchner (2014, 2019)

    Inputs
    ------
    params: dict. Dictionary of input parameters.  Must include an entry for 
                  kll, loglike, model, outputdir, pnames, and prior.
                  See user manual for descriptions.

    Outputs
    -------
    outp : array. Each set of accepted parameter values.
    bestp: array. Best parameter values.
    """
    # Setup the sampler
    sampler = ultranest.ReactiveNestedSampler(params["pnames"], 
                                              params["loglike"], 
                                              params["prior"], 
                                              log_dir=params["outputdir"], 
                                              vectorized=True)
    # Run it
    out = sampler.run(min_ess=params["min_ess"], max_iters=params["max_iters"], 
                      min_num_live_points=params["min_num_live_points"])
    sampler.print_results()

    # Posterior and best parameters
    bestp = np.array(sampler.results['maximum_likelihood']['point'])
    outp  = sampler.results['samples']
    if params["kll"] is not None:
        for i in range(outp.shape[0]):
            params["kll"].update(params["model"](outp[i], fullout=True))
    outp = outp.T

    # Plotting
    sampler.plot_corner()
    sampler.plot_run()
    sampler.plot_trace()

    return outp, bestp


