#! /usr/bin/env python

import sys, os
import functools
import numpy as np

import dream_func as df
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

func = functools.partial(df.model, x=x)

loglike = functools.partial(df.loglikelihood, 
                            data=data, uncert=uncert, model=func)

loglike.__name__ = 'loglike'

# Ensure the output directory exists
outputdir = "./output_dream/"
if not os.path.isdir(outputdir):
    os.mkdir(outputdir)

# Run it
samp = lisa.run('dream', fbestp='output_bestp.npy', fext='.png', 
                         fprefix='model', fsavefile='output_posterior.npy', 
                         loglike=loglike, nchains=4, niter=100000, 
                         outputdir=outputdir, pmax=pmax, pmin=pmin, pnames=pnames, 
                         pstep=pstep, thinning=1, truepars=pars, verb=1)

