"""
Wrapper for UltraNest algorithm of Buchner (2014, 2016, 2019).

driver(): executes the inference
"""

import sys, os
import numpy as np
import matplotlib.pyplot as plt
import ultranest

from helper import BaseSampler


class Sampler(BaseSampler):
    def __init__(self, dlogz=0.1, fbestp='bestp.npy', fext='.png', 
                       frac_remain=0.01, fsavefile='output.npy', kll=None, 
                       Lepsilon=0.001, loglike=None, min_ess=500, model=None, 
                       niter=None, nlive=500, outputdir=None, pnames=None, 
                       prior=None, pstep=None, truepars=None, verb=0):
        # Instantiate attributes from BaseSampler
        super(Sampler, self).__init__()
        # General info about the algorithm
        self.alg = 'ultranest' #name
        self.reqpar = ['loglike', 'model', 'nlive', 'outputdir', 
                       'pnames', 'prior', 'pstep'] #required parameters
        self.optpar = ['dlogz', 'fbestp', 'fext', 'frac_remain', 'fsavefile', 
                       'kll', 'Lepsilon', 'min_ess', 'niter', 
                       'truepars', 'verb'] #optional parameters
        # Only keep help entries relevant to this algorithm
        self.helpinfo = {key : self.helpinfo[key] 
                         for key in self.reqpar+self.optpar}
        # Load supplied parameters
        self.dlogz       = dlogz
        self.fbestp      = fbestp
        self.fext        = fext
        self.frac_remain = frac_remain
        self.fsavefile   = fsavefile
        self.kll         = kll
        self.Lepsilon    = Lepsilon
        self.loglike     = loglike
        self.min_ess     = min_ess
        self.model       = model
        self.niter       = niter
        self.nlive       = nlive
        self.outputdir   = outputdir
        self.pnames      = pnames
        self.prior       = prior
        self.pstep       = pstep
        self.truepars    = truepars
        self.verb        = verb
        if self.verb:
            print("PyMultiNest sampler initialized")
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
        self.check_nonnegfloat('frac_remain')
        self.check_nonnegfloat('Lepsilon')
        # Check positive int inputs
        self.check_posint('min_ess')
        self.check_posint('nlive')
        # Check that required arguments are not none
        self.check_none('loglike')
        self.check_none('model')
        self.check_none('outputdir')
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
            un = ultranest.ReactiveNestedSampler(list(self.pnames), 
                                                 self.loglike, 
                                                 self.prior, 
                                                 log_dir=self.outputdir, 
                                                 vectorized=True)
            # Run it
            out = un.run(min_ess=self.min_ess, max_iters=self.niter, 
                         min_num_live_points=self.nlive, 
                         frac_remain=self.frac_remain, 
                         Lepsilon=self.Lepsilon, dlogz=self.dlogz)
            if self.verb:
                un.print_results()

            # Posterior and best parameters
            self.bestp = np.array(un.results['maximum_likelihood']['point'])
            self.outp  = un.results['samples']
            if self.kll is not None:
                for i in range(outp.shape[0]):
                    self.kll.update(self.model(self.outp[i], fullout=True))
            self.outp = self.outp.T

            # UltraNest Plotting
            un.plot_corner()
            un.plot_run()
            un.plot_trace()

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



def driver(params):
    """
    UltraNest algorithm of Buchner (2014, 2019)

    Inputs
    ------
    params: dict. Dictionary of input parameters.  Must include an entry for 
                  kll, loglike, model, outputdir, pnames, and prior.
                  See user manual for descriptions.

    Outputs
    -------
    outp : array. Each set of accepted parameter values.
    bestp: array. Best parameter values.
    """
    # Setup the sampler
    sampler = ultranest.ReactiveNestedSampler(params["pnames"], 
                                              params["loglike"], 
                                              params["prior"], 
                                              log_dir=params["outputdir"], 
                                              vectorized=True)
    # Run it
    out = sampler.run(min_ess=params["min_ess"], max_iters=params["max_iters"], 
                      min_num_live_points=params["min_num_live_points"], 
                      frac_remain=params["frac_remain"], 
                      Lepsilon=params["Lepsilon"])
    sampler.print_results()

    # Posterior and best parameters
    bestp = np.array(sampler.results['maximum_likelihood']['point'])
    outp  = sampler.results['samples']
    if params["kll"] is not None:
        for i in range(outp.shape[0]):
            params["kll"].update(params["model"](outp[i], fullout=True))
    outp = outp.T

    # Plotting
    sampler.plot_corner()
    sampler.plot_run()
    sampler.plot_trace()

    return outp, bestp


