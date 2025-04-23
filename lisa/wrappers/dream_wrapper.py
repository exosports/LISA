"""
Wrapper for DREAM algorithm of ter Laloy & Vrugt (2012)
as implemented in PyDREAM (https://github.com/LoLab-VU/PyDREAM)

Sampler: class to setup and run the inference
"""

import sys, os
import numpy as np

import scipy.stats as ss

from .helper import BaseSampler
from pydream.parameters import SampledParam
from pydream.core import run_dream
#for Gelman et al convergence tests
mc3dir = os.path.join(os.path.dirname(__file__), '..', 'modules', 'MCcubed')
sys.path.append(mc3dir)
import MCcubed as mc3



class Sampler(BaseSampler):
    def __init__(self, burnin=None, fbestp='output_bestp.npy', fext='.png', 
                       fprefix='model', fsavefile='output_posterior.npy', 
                       loglike=None, multitry=5, nchains=3, niter=None, 
                       outputdir=None, pmax=None, pmin=None, pnames=None, 
                       pstep=None, resume=False, thinning=1, truepars=None, verb=0):
        """
        For details on the inputs, instantiate an object `obj` and call 
        obj.help('parameter'), or see the description in the user manual.
        """
        # Instantiate attributes from BaseSampler
        super(Sampler, self).__init__()
        # General info about the algorithm
        self.alg = 'dream' #name
        self.reqpar = ['loglike', 'nchains', 'niter', 'outputdir', 
                       'pmax', 'pmin'] # required parameters
        self.optpar = ['burnin', 'fbestp', 'fext', 'fprefix', 'fsavefile', 
                       'multitry', 'pnames', 'pstep', 'resume', 'thinning', 
                       'truepars', 'verb'] #optional
        # Only keep help entries relevant to this algorithm
        self.helpinfo = {key : self.helpinfo[key] 
                         for key in self.reqpar+self.optpar}
        # Load supplied parameters
        if burnin is None:
            self.burnin = niter // 2
        else:
            self.burnin = burnin
        self.fbestp     = fbestp
        self.fext       = fext
        self.fsavefile  = fsavefile
        self.loglike    = loglike
        self.fprefix    = fprefix
        self.multitry   = multitry
        self.nchains    = nchains
        self.niter      = niter
        self.outputdir  = outputdir
        self.pmax       = pmax
        self.pmin       = pmin
        self.pnames     = pnames
        if pstep is None:
            self.pstep  = np.ones(pmax.size)
        else:
            self.pstep  = pstep
        self.resume     = resume
        self.thinning   = thinning
        self.truepars   = truepars
        self.verb       = verb
        if self.verb:
            print("DREAM sampler initialized")
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
        self.prep_arr('pmax')
        self.prep_arr('pmin')
        self.prep_arr('pstep')
        # Check positive integers
        self.check_posint('nchains')
        self.check_posint('niter')
        self.check_posint('thinning')
        # Check non-negative integers
        self.check_nonnegint('burnin')
        # Check that required arguments are not none
        self.check_none('loglike')
        # Make sure outputdir is an absolute path & exists
        if self.make_abspath('outputdir'):
            # Now update paths based on that, if needed
            self.update_path('fbestp')
            self.update_path('fprefix')
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
            if self.resume:
                history_file = self.fprefix + '_DREAM_chain_history.npy'
            else:
                history_file = False
            # Run the MCMC
            history, log_ps = run_dream([SampledParam(ss.uniform, 
                                                      loc=self.pmin, scale=self.pmax-self.pmin)], 
                                        self.loglike, niterations=self.niter, nchains=self.nchains, 
                                        start_random=True, save_history=True, 
                                        history_file=history_file, 
                                        multitry=self.multitry, model_name=self.fprefix, 
                                        verbose=self.verb)
            history    = np.asarray(history).swapaxes(1,2)
            log_ps     = np.asarray(log_ps)
            ibest      = np.where(log_ps == log_ps.max())
            self.bestp = history[ibest[0][0], :, ibest[1][0]]
            # Convergence criteria
            try:
                convergence = mc3.mc.convergetest(history)
                print("Rhat convergence metric:")
                print(convergence)
            except Exception as e:
                print("Unable to determine Rhat convergence metric:")
                print(e)
            # Sample size
            try:
                speis, ess = mc3.mc.ess(history)
                print("Steps per effective independent sample:")
                print(speis)
                print("Effective sample size:")
                print(ess)
            except Exception as e:
                print("Unable to determine effective sample size:")
                print(e)
            # Stack the posterior
            self.outp = history[0, :, self.burnin:]
            for i in range(1, self.nchains):
                self.outp = np.hstack((self.outp, history[i, :, self.burnin:]))
            # Save posterior and bestfit params
            if self.fsavefile is not None:
                np.save(self.fsavefile, self.outp)
            if self.fbestp is not None:
                np.save(self.fbestp, self.bestp)
        else:
            if self.verb:
                print("Sampler is not fully prepared to run. " + \
                      "Correct the above errors and try again.")


