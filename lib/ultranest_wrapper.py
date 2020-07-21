"""
Wrapper for UltraNest algorithm of Buchner (2014, 2019).

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
    params: dict. Dictionary of input parameters.

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
                                              vectorized=False)
    # Run it
    out = sampler.run()
    sampler.print_results()

    # Posterior and best parameters
    outp  = sampler.results['samples'].T
    bestp = sampler.results['maximum_likelihood']['point']

    # Plotting
    sampler.plot_corner()
    sampler.plot_run()
    sampler.plot_trace()

    return outp, bestp


