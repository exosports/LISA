#! /usr/bin/env python

import sys, os
import numpy as np


# Read parameters
if len(sys.argv[1:]) != 3:
    raise ValueError("Too many inputs given.  Expected 3.\nReceived: " + \
                     str(sys.argv[1:]))
a, b, c = [float(val) for val in sys.argv[1:]]

# Define the x-axis grid
x = np.arange(-5, 6) # [-5, 5] in steps of 1

# Compute the true data
true = a*x**2 + b*x + c
# get noise for each data point, add it
uncert = np.sqrt(true)
data   = true + np.random.normal(0, uncert)

# Save it
np.save('true.npy'  , true)
np.save('data.npy'  , data)
np.save('uncert.npy', uncert)
np.save('params.npy', np.array([a, b, c]))


