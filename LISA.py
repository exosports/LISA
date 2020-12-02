"""
Main driver for LISA, the Large-selection Interface for Sampling Algorithms

run(): execute a call to a sampler
"""

import sys, os
import numpy as np

mcpdir = os.path.join(os.path.dirname(__file__), 'modules', 'MCcubed', 
                                                 'MCcubed', 'plots')
sys.path.append(mcpdir)
import mcplots as mcp

sys.path.append(os.path.join(os.path.dirname(__file__), 'lib'))
# driver imported below in run()


def run(alg, params):
    """
    Main driver for LISA.

    Inputs
    ------
    alg   : string. Sampling algorithm to use.
                    Options: demc, snooker, multinest, ultranest
    params: dict.   Parameters for the sampling algorithm.
                    For a list & description of parameters, see the user manual.

    Outputs
    -------
    out: Output from the algorithm.  
         See algorithm's documention in lib/ for details.

    Examples
    --------
    See the example/ directory.
    
    """
    # Check input types are correct
    if type(alg) != str:
        raise Exception("alg argument must be a string. " + \
                        "See docstring or user manual for options.")
    if type(params) != dict:
        raise Exception("params argument must be a dictionary. " + \
                        "See user manual for dictionary keys.")

    # Add defaults if not given
    if 'thinning' not in params.keys():
        params['thinning'] = 1
    if 'nchains' not in params.keys():
        params['nchains'] = 1
    if 'savefile' not in params.keys():
        params['savefile'] = ''
    if 'truepars' not in params.keys():
        params['truepars'] = None
    if 'kll' not in params.keys():
        params['kll'] = None
    if 'min_ess' not in params.keys():
        params['min_ess'] = 400
    if 'min_num_live_points' not in params.keys():
        params['min_num_live_points'] = 400
    if 'max_iters' not in params.keys():
        params['max_iters'] = None
    if 'frac_remain' not in params.keys():
        params['frac_remain'] = 0.01
    if 'Lepsilon' not in params.keys():
        params['Lepsilon'] = 0.001

    # Import relevant driver -- this avoids needing to load all of them at once
    if alg == 'demc':
        from demc_wrapper import driver
    elif alg == 'snooker':
        from snooker_wrapper import driver
    elif alg == 'multinest':
        from multinest_wrapper import driver
    elif alg == 'ultranest':
        from ultranest_wrapper import driver
    else:
        raise ValueError("The supplied algorithm does not exist in LISA.\n" + \
                         "Options: demc, snooker, multinest, ultranest\n"   + \
                         "Received: " + alg)

    # Run the inference
    outp, bestp = driver(params)

    # Produce plots
    pnames = np.asarray(params['pnames'])

    mcp.trace(outp, parname=pnames[params['pstep']>0], thinning=params['thinning'], 
              sep=np.size(outp[0]//params['nchains']), 
              savefile=os.path.join(params['outputdir'], 
                                    params['savefile']+"trace.png"),
              truepars=params['truepars'])
    mcp.histogram(outp, parname=pnames[params['pstep']>0], thinning=params['thinning'], 
                  savefile=os.path.join(params['outputdir'], 
                                        params['savefile']+"posterior.png"),
                  truepars=params['truepars'])
    mcp.pairwise(outp, parname=pnames[params['pstep']>0], thinning=params['thinning'], 
                 savefile=os.path.join(params['outputdir'], 
                                       params['savefile']+"pairwise.png"),
                 truepars=params['truepars'])

    return outp, bestp


