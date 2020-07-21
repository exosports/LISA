"""
Wrapper for DEMC algorithm of ter Braak (2006)

driver(): executes the inference
"""

import sys, os
import numpy as np

mc3dir = os.path.join(os.path.dirname(__file__), '..', 'modules', 'MCcubed')
sys.path.append(mc3dir)
import MCcubed as mc3
mcpdir = os.path.join(mc3dir, 'MCcubed', 'plots')
sys.path.append(mcpdir)
import mcplots as mcp


def driver(params):
    """
    DEMC algorithm of ter Braak (2006)

    Inputs
    ------
    params: dict. Dictionary of input parameters.

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
                              walk      = 'demc', 
                              plots     = False, 
                              leastsq   = False, 
                              log       = logfile, 
                              savefile  = params['fsavefile'],
                              savemodel = params['fsavemodel'])

    if params['flog'] is not None:
        logfile.close()

    pnames = np.asarray(params['pnames'])
    mcp.trace(outp, parname=pnames[params['pstep']>0], thinning=params['thinning'], 
              sep=np.size(outp[0]//params['nchains']), 
              savefile=params['outputdir']+params['savefile']+"MCMC_trace.png")
    mcp.histogram(outp, parname=pnames[params['pstep']>0], thinning=params['thinning'], 
                  savefile=params['outputdir']+params['savefile']+"MCMC_posterior.png")
    mcp.pairwise(outp, parname=pnames[params['pstep']>0], thinning=params['thinning'], 
                 savefile=params['outputdir']+params['savefile']+"MCMC_pairwise.png")

    return outp, bestp


