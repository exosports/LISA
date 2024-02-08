"""
Wrapper for the polychord algorithm of Handley et al. (2015a, 2015b).

Sampler: class to setup and run an inference
"""

import sys, os
import numpy as np
import matplotlib.pyplot as plt
import pypolychord
from pypolychord.settings import PolyChordSettings

from .helper import BaseSampler


class Sampler(BaseSampler):
    def __init__(self, dlogz=0.1, dumper=None, fbestp='bestp.npy', 
                       fext='.png', fprefix='run1', fsavefile='output.npy', 
                       kll=None, loglike=None, model=None, nlive=500, 
                       nrepeat=None, outputdir=None, pnames=None, prior=None, 
                       pstep=None, resume=False, truepars=None, verb=0):
        """
        For details on the inputs, instantiate an object `obj` and call 
        obj.help('parameter'), or see the description in the user manual.
        """
        # Instantiate attributes from BaseSampler
        super(Sampler, self).__init__()
        # General info about the algorithm
        self.alg = 'polychord' #name
        self.reqpar = ['loglike', 'model', 'nlive', 'outputdir', 
                       'prior', 'pstep'] #required parameters
        self.optpar = ['dlogz', 'dumper', 'fbestp', 'fext', 'fprefix', 
                       'fsavefile', 'kll', 'nrepeat', 'pnames', 'resume', 
                       'truepars', 'verb'] #optional parameters
        # Only keep help entries relevant to this algorithm
        self.helpinfo = {key : self.helpinfo[key] 
                         for key in self.reqpar+self.optpar}
        # Load supplied parameters
        self.dlogz       = dlogz
        self.dumper      = dumper
        self.fbestp      = fbestp
        self.fext        = fext
        self.fprefix     = fprefix
        self.fsavefile   = fsavefile
        self.kll         = kll
        self.loglike     = loglike
        self.model       = model
        self.nlive       = nlive
        self.nrepeat     = nrepeat
        self.outputdir   = outputdir
        self.pnames      = pnames
        self.prior       = prior
        self.pstep       = pstep
        self.resume      = resume
        self.truepars    = truepars
        self.verb        = verb
        if self.verb:
            print("polychord sampler initialized")
            print("To view a list of required parameters, print obj.reqpar")
            print("To view a list of optional parameters, print obj.optpar")
            print("For details on any parameter, call " + \
                  "obj.help('parameter') or print obj.helpinfo['parameter']")

    def prepare(self):
        """
        Checks that all required parameters are supplied, and loads any 
        supplied binary files
        """
        self.unprepared = 0
        # Prepare inputs that may be arrays
        self.prep_arr('pstep')
        # Check non-negative float inputs
        self.check_nonnegfloat('dlogz')
        # Check positive int inputs
        self.check_posint('nlive')
        # Check that required arguments are not none
        self.check_none('loglike')
        self.check_none('model')
        self.check_none('prior')
        # Make sure outputdir is an absolute path & exists
        if self.make_abspath('outputdir'):
            if os.sep in self.fprefix:
                self.make_dir(os.path.join(self.outputdir, 
                                           self.fprefix.rsplit(os.sep, 1)[0]))
            # Now update paths based on that, if needed
            self.update_path('fbestp')
            self.update_path('fsavefile')
        # Ensure proper pnames exist as numpy array
        self.check_pnames()
        # Ready to run?
        if self.unprepared:
            print("Correct the", self.unprepared, 
                  "issues above, and try again.")
            return False
        else:
            if self.verb:
                print("Sampler successfully prepared to run.")
            return True

    def run(self):
        """
        Executes the inference
        """
        if self.prepare():
            # Setup the inference
            ndim = np.sum(self.pstep > 0)
            settings = PolyChordSettings(ndim, 0)
            settings.base_dir    = self.outputdir
            settings.file_root   = self.fprefix
            settings.nlive       = self.nlive
            settings.read_resume = self.resume
            if self.nrepeat is not None:
                settings.num_repeat      = self.nrepeat
            settings.precision_criterion = self.dlogz
            settings.grade_dims  = [int(ndim)]
            settings.read_resume = False
            settings.feedback    = self.verb
            # Run it
            if self.dumper is not None:
                out = pypolychord.run_polychord(self.loglike, ndim, 0, 
                                                settings, self.prior, 
                                                self.dumper)
            else:
                out = pypolychord.run_polychord(self.loglike, ndim, 0, 
                                                settings, self.prior)

            outp = np.loadtxt(os.path.join(self.outputdir, self.fprefix) +\
                                   '_equal_weights.txt')
            self.outp  = outp[:, 2:].T
            ibest      = np.argmin(outp[:,1])
            self.bestp = self.outp[:,ibest]
            # Save posterior and bestfit params
            if self.fsavefile is not None:
                np.save(self.fsavefile, self.outp)
            if self.fbestp is not None:
                np.save(self.fbestp, self.bestp)
            return self.outp, self.bestp
        else:
            if self.verb:
                print("Sampler is not fully prepared to run. " + \
                      "Correct the above errors and try again.")


