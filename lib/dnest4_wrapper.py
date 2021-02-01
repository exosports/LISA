"""
Wrapper for DNest4 algorithm of Brewer & Foreman-Mackey (2018).

DNest4_Model: class used as input for DNest4

Sampler: class to setup and run an inference
"""

import sys, os
import numpy as np
import matplotlib.pyplot as plt
import dnest4

from helper import BaseSampler


class DNest4_Model(object):
    def __init__(self, loglike=None, perturb=None, prior=None):
        self.log_likelihood = loglike
        self.perturb        = perturb
        self.from_prior     = prior


class Sampler(BaseSampler):
    def __init__(self, beta=100, fbestp='bestp.npy', fext='.png', 
                       fsavefile='output.npy', kll=None, lam=5, 
                       loglike=None, model=None, niter=None, 
                       nlevel=30, nlevelint=10000, nperstep=10000, 
                       outputdir=None, perturb=None, pnames=None, 
                       prior=None, pstep=None, resample=100, truepars=None, 
                       verb=0):
        # Instantiate attributes from BaseSampler
        super(Sampler, self).__init__()
        # General info about the algorithm
        self.alg = 'dnest4' #name
        self.reqpar = ['loglike', 'model', 'niter', 'nlevel', 'nlevelint', 
                       'nperstep', 'outputdir', 'perturb', 
                       'prior', 'pstep'] #required parameters
        self.optpar = ['beta', 'fbestp', 'fext', 'fsavefile', 
                       'kll', 'lam', 'pnames', 'resample', 
                       'truepars', 'verb'] #optional parameters
        # Only keep help entries relevant to this algorithm
        self.helpinfo = {key : self.helpinfo[key] 
                         for key in self.reqpar+self.optpar}
        # Load supplied parameters
        self.beta        = beta
        self.fbestp      = fbestp
        self.fext        = fext
        self.fsavefile   = fsavefile
        self.kll         = kll
        self.lam         = lam
        self.loglike     = loglike
        self.model       = model
        self.niter       = niter
        self.nlevel      = nlevel
        self.nlevelint   = nlevelint
        self.nperstep    = nperstep
        self.outputdir   = outputdir
        self.perturb     = perturb
        self.pnames      = pnames
        self.prior       = prior
        self.pstep       = pstep
        self.resample    = resample
        self.truepars    = truepars
        self.verb        = verb
        if self.verb:
            print("DNest4 sampler initialized")
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
        # Check positive int inputs
        self.check_posint('nlevel')
        self.check_posint('nlevelint')
        self.check_posint('nperstep')
        # Check non-negative floats
        self.check_nonnegfloat('beta')
        self.check_nonnegfloat('lam')
        self.check_nonnegfloat('resample')
        # Check that required arguments are not none
        self.check_none('loglike')
        self.check_none('model')
        self.check_none('niter')
        self.check_none('outputdir')
        self.check_none('perturb')
        self.check_none('prior')
        # Make sure outputdir is an absolute path & exists
        self.make_abspath('outputdir')
        self.make_dir(self.outputdir)
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
            # Set up the inference
            backend = dnest4.backends.CSVBackend(basedir=self.outputdir, 
                                                 sep=" ")
            dns = dnest4.DNest4Sampler(DNest4_Model(loglike=self.loglike, 
                                                    perturb=self.perturb, 
                                                    prior=self.prior), 
                                       backend=backend)
            # Run it
            out = dns.sample(max_num_levels=self.nlevel, 
                             num_steps=self.niter, 
                             new_level_interval=self.nlevelint, 
                             num_per_step=self.nperstep, lam=self.lam, 
                             beta=self.beta)
            for i, samp in enumerate(out):
                if self.verb:
                    print(''.join(['Iteration: ', str(i+1), '/', 
                                   str(self.niter)]), end='\r')
            print('')
            # Best-fit parameters
            ibest      = np.argmax(backend.sample_info["log_likelihood"])
            self.bestp = backend.samples[ibest]
            # Resample for posterior
            stats = dns.postprocess(resample=self.resample)
            self.outp = backend.posterior_samples.T
            """
            stats = dns.postprocess()
            # Load the weights to properly estimate the posterior
            #weights = np.loadtxt(os.path.join(self.outputdir, 'weights.txt'))
            weights = np.squeeze(backend.weights)
            wsh     = weights.shape[0]
            # Remove cases w/ 0 weight
            samps   = backend.samples
            samps   = samps  [weights!=0]
            weights = weights[weights!=0]
            # As in MultiNest, resample to equal weights by considering 
            # int(nsamples*weight) repetitions
            self.outp = []
            for i in range(len(weights)):
                self.outp.extend([samps[i] 
                                  for j in range(int(wsh * weights[i]))])
            self.outp = np.asarray(self.outp).T
            """
            # Quantiles
            if self.kll is not None:
                for i in range(self.outp.shape[-1]):
                    self.kll.update(self.model(self.outp[:,i], fullout=True))
            # Save posterior and bestfit params
            if self.fsavefile is not None:
                np.save(self.fsavefile, self.outp)
            if self.fbestp is not None:
                np.save(self.fbestp, self.bestp)
        else:
            if self.verb:
                print("Sampler is not fully prepared to run. " + \
                      "Correct the above errors and try again.")


