"""
Wrapper for DEMC with snooker updating algorithm of ter Braak & Vrugt (2008)

driver(): executes the inference
"""

import sys, os
import numpy as np

mc3dir = os.path.join(os.path.dirname(__file__), '..', 'modules', 'MCcubed')
sys.path.append(mc3dir)
import MCcubed as mc3


def driver(params):
    """
    DEMC with snooker updating algorithm of ter Braak & Vrugt (2008)

    Inputs
    ------
    params: dict. Dictionary of input parameters.  Must include an entry for 
                  burnin, data, flog, fsavefile, fsavemodel, func, hsize, 
                  indparams, nchains, niter, outputdir, pinit, pmax, pmin, 
                  pnames, pstep, savefile, thinning, and uncert.
                  See user manual for descriptions.

    Outputs
    -------
    outp : array. Each set of accepted parameter values.
    bestp: array. Best parameter values.
    """
    if params['flog'] is not None:
        logfile = open(params['flog'], 'w')
    else:
        logfile = None

    # Run the MCMC
    outp, bestp = mc3.mc.mcmc(params['data'], params['uncert'], 
                              func      = params['func'], 
                              indparams = params['indparams'],
                              parnames  = params['pnames'], 
                              params    = params['pinit'], 
                              pmin      = params['pmin'], 
                              pmax      = params['pmax'], 
                              stepsize  = params['pstep'],
                              numit     = params['niter'], 
                              burnin    = params['burnin'], 
                              thinning  = params['thinning'], 
                              nchains   = params['nchains'], 
                              walk      = 'snooker', 
                              hsize     = params['hsize'], 
                              plots     = False, 
                              leastsq   = False, 
                              log       = logfile, 
                              savefile  = params['fsavefile'],
                              savemodel = params['fsavemodel'])

    if params['flog'] is not None:
        logfile.close()

    return outp, bestp


