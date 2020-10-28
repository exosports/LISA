#! /usr/bin/env python

import sys, os
import functools
import numpy as np

import ultranest_func as unf
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

func = functools.partial(unf.model, x=x)

prior = functools.partial(unf.prior, pmin=pmin, pmax=pmax, pstep=pstep)

loglike = functools.partial(unf.loglikelihood, 
                            data=data, uncert=uncert, model=func)

prior.__name__ = 'prior'
loglike.__name__ = 'loglike'

# Ensure the output directory exists
outputdir = "./output_ultranest/"
if not os.path.isdir(outputdir):
    os.mkdir(outputdir)

# Parameter dictionary for LISA
params = {"prior"  : prior , "loglike" : loglike, 
          "pnames" : pnames, "pstep"   : pstep  , 
          "kll"    : None  , "model"   : func   , 
          "outputdir" : outputdir, 
          "truepars"  : pars}

# Run it
outp, bestp = LISA.run('ultranest', params)

