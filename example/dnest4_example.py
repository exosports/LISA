#! /usr/bin/env python

import sys, os
import functools
import numpy as np

import dnest4_func as dnf
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

func = functools.partial(dnf.model, x=x)

prior = functools.partial(dnf.prior, ndim=np.sum(pstep>0), 
                          pmin=pmin[pstep>0], pmax=pmax[pstep>0])

loglike = functools.partial(dnf.loglikelihood, 
                            data=data, uncert=uncert, model=func)

perturb = functools.partial(dnf.perturb, ndim=np.sum(pstep>0), 
                            width=pmax[pstep>0]-pmin[pstep>0])

prior.__name__   = 'prior'
loglike.__name__ = 'loglike'
perturb.__name__ = 'perturb'

# Ensure the output directory exists
outputdir = "./output_dnest4/"
if not os.path.isdir(outputdir):
    os.mkdir(outputdir)

# Run it
samp = LISA.run('dnest4', fbestp='output_bestp.npy', 
                fext='.png', fsavefile='output_posterior.npy', 
                kll=None, loglike=loglike, model=func, 
                niter=100000, nlevel=40, nlevelint=10000, nperstep=100, 
                outputdir=outputdir, perturb=perturb, 
                pnames=pnames, prior=prior, pstep=pstep, 
                truepars=pars, verb=1)


