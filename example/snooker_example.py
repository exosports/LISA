#! /usr/bin/env python

import sys, os
import functools
import numpy as np

from mcmc_func import model
# Works whether the run directory is within the LISA repo, or parallel to it
# If executing from a different location than either of those, users will 
# need to adjust the following sys.path.append accordingly.
sys.path.append('../')
sys.path.append('../LISA/')
import LISA


# Load the inputs, and true parameters
data   = np.load('data.npy')
uncert = np.load('uncert.npy')
pars   = np.load('params.npy')

pnames = ['a', 'b', 'c']              # Parameter names
pinit  = np.array([  0.,   0.,   0.]) # Initial param values
pmin   = np.array([-10., -10., -10.]) # Minimum allowed values
pmax   = np.array([ 10.,  10.,  10.]) # Maximum allowed values
pstep  = np.array([  3.,   3.,   3.]) # "Step" size (used for initial samples)

# The defined x-axis values corresponding to the data
x = np.arange(-5, 6)[None, :]

func = functools.partial(model, x=x)

# Ensure the output directory exists
outputdir = "./output_snooker/"
if not os.path.isdir(outputdir):
    os.mkdir(outputdir)

# Run it
samp = LISA.run('snooker', burnin=4000, data=data, 
                fbestp='output_bestp.npy', 
                fext='.png', flog='MCMC.log', 
                fsavefile='output_posterior.npy', fsavemodel=None, 
                hsize=100, indparams=[], kll=None, model=func, 
                modelper=0, nchains=10, niter=100000, 
                outputdir=outputdir, pinit=pinit, pmax=pmax, 
                pmin=pmin, pnames=pnames, pstep=pstep, thinning=1, 
                truepars=pars, uncert=uncert, verb=1)


