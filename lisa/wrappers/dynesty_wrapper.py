"""
Wrapper for dynesty dynamic nested sampling algorithm of Speagle (2019).

Sampler: class to setup and run an inference
"""

import sys, os
import multiprocess as mp
import numpy as np
import matplotlib.pyplot as plt

import dynesty

from .helper import BaseSampler


class Sampler(BaseSampler):
    def __init__(self, bound='multi', dlogz=0.1, fbestp='bestp.npy', 
                       fext='.png', fsavefile='output.npy', kll=None, 
                       loglike=None, min_ess=500, model=None, nchains=1, 
                       niter=None, nlive=500, nlive_batch=500, outputdir=None, 
                       pnames=None, prior=None, pstep=None, sample='auto', 
                       truepars=None, fcheckpoint='dynesty.save', resume=False, verb=0):
        """
        For details on the inputs, instantiate an object `obj` and call 
        obj.help('parameter'), or see the description in the user manual.
        """
        # Instantiate attributes from BaseSampler
        super(Sampler, self).__init__()
        # General info about the algorithm
        self.alg = 'dynesty' #name
        self.reqpar = ['loglike', 'model', 'nlive', 'nlive_batch', 'outputdir', 
                       'prior', 'pstep'] #required parameters
        self.optpar = ['bound', 'dlogz', 'fbestp', 'fext', 'fsavefile', 
                       'kll', 'min_ess', 'niter', 'pnames', 'sample', 
                       'truepars', 'fcheckpoint', 'resume', 'verb'] #optional parameters
        # Only keep help entries relevant to this algorithm
        self.helpinfo = {key : self.helpinfo[key] 
                         for key in self.reqpar+self.optpar}
        # Load supplied parameters
        self.bound       = bound
        self.dlogz       = dlogz
        self.fbestp      = fbestp
        self.fext        = fext
        self.fsavefile   = fsavefile
        self.kll         = kll
        self.loglike     = loglike
        self.min_ess     = min_ess
        self.model       = model
        self.nchains     = nchains
        self.niter       = niter
        self.nlive       = nlive
        self.nlive_batch = nlive_batch
        self.outputdir   = outputdir
        self.pnames      = pnames
        self.prior       = prior
        self.pstep       = pstep
        self.sample      = sample
        self.truepars    = truepars
        self.fcheckpoint = fcheckpoint
        self.resume      = resume
        self.verb        = verb
        if self.verb:
            print("dynesty sampler initialized")
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
        self.check_posint('min_ess')
        self.check_posint('nchains')
        self.check_posint('nlive')
        self.check_posint('nlive_batch')
        # Check that required arguments are not none
        self.check_none('loglike')
        self.check_none('model')
        self.check_none('prior')
        # Make sure outputdir is an absolute path & exists
        if self.make_abspath('outputdir'):
            # Now update paths based on that, if needed
            self.update_path('fbestp')
            self.update_path('fsavefile')
            self.update_path('fcheckpoint')
            if not self.resume and os.path.exists(self.fcheckpoint):
                print("Resume is set to False, but the specified checkpoint file already exists.")
                self.unprepared += 1
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
            if self.nchains > 1:
                p = mp.Pool(self.nchains)
                queue_size = self.nchains
            else:
                p = None
                queue_size = 1
            if not self.resume:
                dy = dynesty.DynamicNestedSampler(self.loglike, 
                                                  self.prior, 
                                                  ndim, 
                                                  bound=self.bound, 
                                                  sample=self.sample, 
                                                  queue_size=self.nchains, 
                                                  pool=p)
            else:
                dy = dynesty.NestedSampler.restore(self.fcheckpoint, pool=p)

            # Run it
            dy.run_nested(nlive_init=self.nlive, 
                          nlive_batch=self.nlive_batch, 
                          maxiter=self.niter, dlogz_init=self.dlogz, 
                          n_effective=self.min_ess, 
                          checkpoint_file=self.fcheckpoint, 
                          resume=self.resume)
            results = dy.results

            # Posterior and best parameters
            self.bestp = results["samples"][np.argmax(results["logl"])]
            samps      = results["samples"]
            nsamp      = samps.shape[0]
            # From cornerplot in dynesty/plotting.py 
            try:
                weights = np.exp(results['logwt'] - results['logz'][-1])
            except:
                weights = results['weights']
            # Remove cases w/ 0 weight
            samps   = samps  [weights!=0]
            weights = weights[weights!=0]

            # As in MultiNest, resample to equal weights by considering 
            # int(nsamples*weight) repetitions
            self.outp = []
            for i in range(len(weights)):
                self.outp.extend([samps[i] 
                                  for j in range(int(nsamp * weights[i]))])
            self.outp = np.asarray(self.outp)

            if self.kll is not None:
                for i in range(self.outp.shape[0]):
                    self.kll.update(self.model(self.outp[i], fullout=True))
            self.outp = self.outp.T

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


