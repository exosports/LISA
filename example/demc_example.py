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
outputdir = "./output_demc/"
if not os.path.isdir(outputdir):
    os.mkdir(outputdir)

# Parameter dictionary for LISA
params = {"data"      : data  , "uncert"    : uncert, 
          "truepars"  : pars  , "func"      : func,
          "indparams" : []    , "pnames"    : pnames, 
          "pinit"     : pinit , "pmin"      : pmin, 
          "pmax"      : pmax  , "pstep"     : pstep, 
          "niter"     : 100000, "burnin"    : 4000, 
          "thinning"  : 1     , "nchains"   : 10, 
          "savefile"  : ""    , "outputdir" : outputdir, 
          "fsavefile" : outputdir+"output.npy", 
          "fsavemodel": "", 
          "flog"      : outputdir+"MCMC.log"}

# Run it
outp, bestp = LISA.run('demc', params)



