"""
Wrapper for PyMultiNest algorithm of Buchner (2014).

driver(): executes the inference
"""

import sys, os
import numpy as np
import matplotlib.pyplot as plt
import pymultinest


def driver(params):
    """
    PyMultiNest algorithm of Buchner (2014)

    Inputs
    ------
    params: dict. Dictionary of input parameters.  Must include an entry for 
                  kll, loglike, model, outputdir, pnames, prior, and pstep.
                  See user manual for descriptions.

    Outputs
    -------
    a : Analyzer object. See PyMultiNest's documentation.
    """
    pnames = np.asarray(params["pnames"])
    sampler = pymultinest.run(params["loglike"], params["prior"], 
                              n_dims=len(pnames[params["pstep"]>0]), 
                              outputfiles_basename=os.path.join(params["outputdir"], 'out'), 
                              resume=False)
    a = pymultinest.Analyzer(n_params=len(params["pnames"]), 
                             outputfiles_basename=os.path.join(params["outputdir"], 'out'))
    s = a.get_stats()
    bestp = a.get_best_fit()['parameters']
    outp  = a.get_equal_weighted_posterior()[:, :-1]
    if params["kll"] is not None:
        for i in range(outp.shape[0]):
            params["kll"].update(params["model"](outp[i], fullout=True))
    outp = outp.T

    print("Global Evidence:\n\t%.15e +- %.15e" % (s['nested sampling global log-evidence'], s['nested sampling global log-evidence error']))

    # Plotting -- taken from pymultinest example(s)
    n_params = outp.shape[0]
    parameters = pnames[params["pstep"]>0]
    p = pymultinest.PlotMarginalModes(a)
    plt.figure(figsize=(5*n_params, 5*n_params))
    #plt.subplots_adjust(wspace=0, hspace=0)
    for i in range(n_params):
        plt.subplot(n_params, n_params, n_params * i + i + 1)
        p.plot_marginal(i, with_ellipses = True, with_points = False, grid_points=50)
        plt.ylabel("Probability")
        plt.xlabel(parameters[i])
        
        for j in range(i):
	        plt.subplot(n_params, n_params, n_params * j + i + 1)
	        #plt.subplots_adjust(left=0, bottom=0, right=0, top=0, wspace=0, hspace=0)
	        p.plot_conditional(i, j, with_ellipses = False, with_points = True, grid_points=30)
	        plt.xlabel(parameters[i])
	        plt.ylabel(parameters[j])

    plt.savefig(os.path.join(params["outputdir"], 'marginals_multinest.pdf'), bbox_inches='tight')
    plt.close()

    # These are optional since the above contains the same info
    for i in range(n_params):
        outfile = '%s-mode-marginal-%d.pdf' % (a.outputfiles_basename,i)
        p.plot_modes_marginal(i, with_ellipses = True, with_points = False)
        plt.ylabel("Probability")
        plt.xlabel(parameters[i])
        plt.savefig(outfile, format='pdf', bbox_inches='tight')
        plt.close()
        
        outfile = '%s-mode-marginal-cumulative-%d.pdf' % (a.outputfiles_basename,i)
        p.plot_modes_marginal(i, cumulative = True, with_ellipses = True, with_points = False)
        plt.ylabel("Cumulative probability")
        plt.xlabel(parameters[i])
        plt.savefig(outfile, format='pdf', bbox_inches='tight')
        plt.close()

    return outp, bestp


