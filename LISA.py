"""
Main driver for LISA, the Large-selection Interface for Sampling Algorithms

run(): execute a call to a sampler
"""

import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), 'lib'))
# driver imported in run()


def run(alg, params):
    """
    Main driver for LISA.

    Inputs
    ------
    alg   : string. Sampling algorithm to use.
                    Options: snooker, demc
    params: dict.   Parameters for the sampling algorithm.
                    For a list of parameters, see the algorithm's documentation
                    in lib/

    Outputs
    -------
    out: Output from the algorithm.  
         See algorithm's documention in lib/ for details.

    Examples
    --------
    
    """
    if type(alg) != str:
        raise Exception("alg argument must be a string. " + \
                        "See docstring for options.")
    if type(params) != dict:
        raise Exception("params argument must be a dictionary. " + \
                        "See alg's documentation in lib/ for options.")

    # Import relevant driver -- this avoids needing to load all of them at once
    if alg == 'demc':
        from demc_wrapper import driver
    elif alg == 'snooker':
        from snooker_wrapper import driver
    elif alg == 'multinest':
        from multinest_wrapper import driver
    elif alg == 'ultranest':
        from ultranest_wrapper import driver

    out = driver(params)

    return out


