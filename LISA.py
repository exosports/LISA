"""
Main driver for LISA, the Large-selection Interface for Sampling Algorithms

run(): execute a call to a sampler
"""

import sys, os
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), 'lib'))
# Sampler imported below in setup()


def setup(alg, **kwargs):
    """
    Initializes the specified sampler.

    Inputs
    ------
    alg     : string. Sampling algorithm to use.
                      Options: demc, dynesty, multinest, snooker, ultranest
    **kwargs: Parameters for the sampling algorithm.
              For a list & description of parameters, see the user manual.

    Outputs
    -------
    Sampler object, with the supplied **kwargs set.
    """
    if alg == 'demc':
        from demc_wrapper      import Sampler
    elif alg == 'dynesty':
        from dynesty_wrapper   import Sampler
    elif alg == 'multinest':
        from multinest_wrapper import Sampler
    elif alg == 'snooker':
        from snooker_wrapper   import Sampler
    elif alg == 'ultranest':
        from ultranest_wrapper import Sampler
    else:
        raise ValueError("The supplied algorithm does not exist in LISA.\n" + \
                         "Options: demc, dynesty, multinest, snooker, " + \
                         "ultranest\nReceived: " + alg)
    return Sampler(**kwargs)


def run(alg, **kwargs):
    """
    Initializes the specified sampler, runs it, and produces output plots.

    Inputs
    ------
    alg     : string. Sampling algorithm to use.
                      Options: demc, dynesty, multinest, snooker, ultranest
    **kwargs: Parameters for the sampling algorithm.
              For a list & description of parameters, see the user manual.

    Outputs
    -------
    samp : Sampler object, with the supplied **kwargs set.
    """
    samp = setup(alg, **kwargs)
    samp.run()
    samp.make_plots()
    return samp


