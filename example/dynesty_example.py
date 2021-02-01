#! /usr/bin/env python

import sys, os
import functools
import numpy as np

import dynesty_func as dyf
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
x = np.arange(-5, 6)

func = functools.partial(dyf.model, x=x)

prior = functools.partial(dyf.prior, pmin=pmin, pmax=pmax, pstep=pstep)

loglike = functools.partial(dyf.loglikelihood, 
                            data=data, uncert=uncert, model=func)

prior.__name__ = 'prior'
loglike.__name__ = 'loglike'

# Ensure the output directory exists
outputdir = "./output_dynesty/"
if not os.path.isdir(outputdir):
    os.mkdir(outputdir)

# Run it
samp = LISA.run('dynesty', dlogz=0.001, fbestp='output_bestp.npy', 
                fext='.png', fsavefile='output_posterior.npy', 
                kll=None, loglike=loglike, model=func, 
                niter=100000, nlive=1000, outputdir=outputdir, 
                pnames=pnames, prior=prior, pstep=pstep, 
                truepars=pars, verb=1)


