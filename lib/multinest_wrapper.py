"""
Wrapper for PyMultiNest algorithm of Buchner (2014).

Sampler: class to setup and run an inference
"""

import sys, os
import numpy as np
import matplotlib.pyplot as plt
import pymultinest

from helper import BaseSampler


class Sampler(BaseSampler):
    def __init__(self, dlogz=0.1, fbestp='bestp.npy', fext='.png', fprefix='pmn/', 
                       fsavefile='output.npy', kll=None, 
                       loglike=None, model=None, niter=0, nlive=500, 
                       outputdir=None, pnames=None, 
                       prior=None, pstep=None, truepars=None, verb=0):
        """
        For details on the inputs, instantiate an object `obj` and call 
        obj.help('parameter'), or see the description in the user manual.
        """
        # Instantiate attributes from BaseSampler
        super(Sampler, self).__init__()
        # General info about the algorithm
        self.alg = 'multinest' #name
        self.reqpar = ['loglike', 'model', 'nlive', 'outputdir', 
                       'prior', 'pstep'] #required parameters
        self.optpar = ['dlogz', 'fbestp', 'fext', 'fprefix', 'fsavefile', 
                       'kll', 'niter', 'pnames', 'truepars', 
                       'verb'] #optional parameters
        # Only keep help entries relevant to this algorithm
        self.helpinfo = {key : self.helpinfo[key] 
                         for key in self.reqpar+self.optpar}
        # Load supplied parameters
        self.dlogz      = dlogz
        self.fbestp     = fbestp
        self.fext       = fext
        self.fprefix    = fprefix
        self.fsavefile  = fsavefile
        self.kll        = kll
        self.loglike    = loglike
        self.model      = model
        self.niter      = niter
        self.nlive      = nlive
        self.outputdir  = outputdir
        self.pnames     = pnames
        self.prior      = prior
        self.pstep      = pstep
        self.truepars   = truepars
        self.verb       = verb
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
        # Check non-negative int inputs
        self.check_nonnegint('niter')
        # Check positive int inputs
        self.check_posint('nlive')
        # Check arguments that cannot be none
        self.check_none('fprefix')
        self.check_none('loglike')
        self.check_none('model')
        self.check_none('outputdir')
        self.check_none('prior')
        # Make sure outputdir is an absolute path & exists
        self.make_abspath('outputdir')
        self.make_dir(self.outputdir)
        if os.sep in self.fprefix:
            self.make_dir(os.path.join(self.outputdir, self.fprefix))
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
            # Run the inference
            pmn = pymultinest.run(self.loglike, self.prior, 
                              n_dims=sum(self.pstep > 0), 
                              outputfiles_basename=os.path.join(self.outputdir, 
                                                                self.fprefix), 
                              n_live_points=self.nlive, max_iter=self.niter, 
                              evidence_tolerance=self.dlogz)
            # Analyze the output
            a = pymultinest.Analyzer(n_params=len(self.pstep), 
                            outputfiles_basename=os.path.join(self.outputdir, 
                                                              self.fprefix))
            s = a.get_stats()
            self.bestp = a.get_best_fit()['parameters']
            self.outp  = a.get_equal_weighted_posterior()[:, :-1]
            # Update quantiles
            if self.kll is not None:
                for i in range(self.outp.shape[0]):
                    self.kll.update(self.model(self.outp[i], fullout=True))
            self.outp = self.outp.T

            if self.verb:
                print("Global Evidence:\n\t%.15e +- %.15e" % \
                      (s['nested sampling global log-evidence'], 
                       s['nested sampling global log-evidence error']))

            # PyMultiNest plots 
            n_params = self.outp.shape[0]
            if self.pnames is not None:
                parameters = self.pnames[self.pstep>0]
            p = pymultinest.PlotMarginalModes(a)
            plt.figure(figsize=(5*n_params, 5*n_params))
            for i in range(n_params):
                plt.subplot(n_params, n_params, n_params * i + i + 1)
                p.plot_marginal(i, with_ellipses = True, with_points = False, 
                                grid_points=50)
                plt.ylabel("Probability")
                plt.xlabel(parameters[i])
                
                for j in range(i):
	                plt.subplot(n_params, n_params, n_params * j + i + 1)
	                p.plot_conditional(i, j, with_ellipses = False, 
                                       with_points = True, grid_points=30)
	                plt.xlabel(parameters[i])
	                plt.ylabel(parameters[j])

            plt.savefig(os.path.join(self.outputdir, ''.join(['marginals_multinest', 
                                     self.fext])), bbox_inches='tight')
            plt.close()

            # These are optional since the above contains the same info
            for i in range(n_params):
                p.plot_modes_marginal(i, with_ellipses = True, with_points = False)
                plt.ylabel("Probability")
                plt.xlabel(parameters[i])
                plt.savefig(''.join([a.outputfiles_basename, 'mode-marginal-', 
                            str(i), self.fext]), bbox_inches='tight')
                plt.close()
                
                p.plot_modes_marginal(i, cumulative = True, with_ellipses = True, with_points = False)
                plt.ylabel("Cumulative probability")
                plt.xlabel(parameters[i])
                plt.savefig(''.join([a.outputfiles_basename, 'mode-marginal-cumulative-', 
                            str(i), self.fext]), bbox_inches='tight')
                plt.close()


            # Save posterior and bestfit params
            if self.fsavefile is not None:
                np.save(self.fsavefile, self.outp)
            if self.fbestp is not None:
                np.save(self.fbestp, self.bestp)
        else:
            if self.verb:
                print("Sampler is not fully prepared to run. " + \
                      "Correct the above errors and try again.")


