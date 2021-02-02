#! /usr/bin/env python

import sys, os
import functools
import numpy as np

import polychord_func as pcf
import lisa


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

func = functools.partial(pcf.model, x=x)

prior = functools.partial(pcf.prior, pmin=pmin[pstep>0], pmax=pmax[pstep>0])

loglike = functools.partial(pcf.loglikelihood, 
                            data=data, uncert=uncert, model=func)

prior.__name__   = 'prior'
loglike.__name__ = 'loglike'

# Ensure the output directory exists
outputdir = "./output_polychord/"
if not os.path.isdir(outputdir):
    os.mkdir(outputdir)

# Run it
samp = lisa.run('polychord', fbestp='output_bestp.npy', 
                fext='.png', fsavefile='output_posterior.npy', 
                kll=None, loglike=loglike, model=func, 
                nlive=500, outputdir=outputdir, 
                pnames=pnames, prior=prior, pstep=pstep, 
                truepars=pars, verb=1)


