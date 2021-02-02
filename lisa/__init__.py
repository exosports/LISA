"""
Main driver for LISA, the Large-selection Interface for Sampling Algorithms

Functions
---------
setup: initialize a sampler
run  : initialize a sampler, execute it, and produce posterior plots
"""

__all__ = ['wrappers']

import sys
import os
import numpy as np

from . import wrappers
from ._version import __version__


def setup(alg, **kwargs):
    """
    Initializes the specified sampler.

    Inputs
    ------
    alg     : string. Sampling algorithm to use.
                      Options: demc, dnest4, dynesty, multinest, polychord, 
                               snooker, ultranest
    **kwargs: Parameters for the sampling algorithm.
              For a list & description of parameters, see the user manual.

    Outputs
    -------
    Sampler object, with the supplied **kwargs set.
    """
    if alg == 'demc':
        #from demc_wrapper      import Sampler
        Sampler = wrappers.demc_wrapper.Sampler
    elif alg == 'dnest4':
        #from dnest4_wrapper    import Sampler
        Sampler = wrappers.dnest4_wrapper.Sampler
    elif alg == 'dynesty':
        #from dynesty_wrapper   import Sampler
        Sampler = wrappers.dynesty_wrapper.Sampler
    elif alg == 'multinest':
        #from multinest_wrapper import Sampler
        Sampler = wrappers.multinest_wrapper.Sampler
    elif alg == 'polychord':
        #from polychord_wrapper import Sampler
        Sampler = wrappers.polychord_wrapper.Sampler
    elif alg == 'snooker':
        #from snooker_wrapper   import Sampler
        Sampler = wrappers.snooker_wrapper.Sampler
    elif alg == 'ultranest':
        #from ultranest_wrapper import Sampler
        Sampler = wrappers.ultranest_wrapper.Sampler
    else:
        raise ValueError("The supplied algorithm does not exist in LISA.\n" + \
                         "Options: demc, dnest4, dynesty, multinest, " + \
                         "polychord, snooker, ultranest\nReceived: " + alg)
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


