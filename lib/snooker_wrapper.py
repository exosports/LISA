"""
Wrapper for DEMC with snooker updating algorithm of ter Braak & Vrugt (2008)

driver(): executes the inference
"""

import sys, os
import numpy as np

from helper import BaseSampler

mc3dir = os.path.join(os.path.dirname(__file__), '..', 'modules', 'MCcubed')
sys.path.append(mc3dir)
import MCcubed as mc3


class Sampler(BaseSampler):
    def __init__(self, burnin=None, data=None, fbestp='bestp.npy', 
                       fext='.png', flog='MCMC.log', 
                       fsavefile='output.npy', 
                       fsavemodel=None, func=None, hsize=1, indparams=[], 
                       kll=None, modelper=0, nchains=1, niter=None, 
                       outputdir=None, 
                       pinit=None, pmax=None, pmin=None, pnames=None, 
                       pstep=None, thinning=1, truepars=None, uncert=None, 
                       verb=0):
        # Instantiate attributes from BaseSampler
        super(Sampler, self).__init__()
        # General info about the algorithm
        self.alg = 'snooker' #name
        self.reqpar = ['burnin', 'data', 'func', 'nchains', 'niter', 
                       'outputdir', 'pinit', 'pmax', 'pmin', 'pstep', 
                       'uncert'] #required parameters
        self.optpar = ['fbestp', 'flog', 'fext', 'fsavefile', 'fsavemodel', 
                       'indparams', 'kll', 'modelper', 'pnames', 'thinning', 
                       'truepars', 'verb'] #optional parameters
        # Only keep help entries relevant to this algorithm
        self.helpinfo = {key : self.helpinfo[key] 
                         for key in self.reqpar+self.optpar}
        # Load supplied parameters
        self.burnin     = burnin
        self.data       = data
        self.fbestp     = fbestp
        self.fext       = fext
        self.flog       = flog
        self.fsavefile  = fsavefile
        self.fsavemodel = fsavemodel
        self.func       = func
        self.hsize      = hsize
        self.indparams  = indparams
        self.kll        = kll
        self.nchains    = nchains
        self.niter      = niter
        self.outputdir  = outputdir
        self.pinit      = pinit
        self.pmax       = pmax
        self.pmin       = pmin
        self.pnames     = pnames
        self.pstep      = pstep
        self.thinning   = thinning
        self.truepars   = truepars
        self.uncert     = uncert
        self.verb       = verb
        if self.verb:
            print("DEMC sampler initialized")
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
        self.prep_arr('data')
        self.prep_arr('pinit')
        self.prep_arr('pmax')
        self.prep_arr('pmin')
        self.prep_arr('pstep')
        self.prep_arr('uncert')
        # Check positive integers
        self.check_posint('hsize')
        self.check_posint('nchains')
        self.check_posint('niter')
        self.check_posint('thinning')
        # Check non-negative integers
        self.check_nonnegint('burnin')
        # Check that required arguments are not none
        self.check_none('func')
        self.check_none('outputdir')
        # Make sure outputdir is an absolute path & exists
        self.make_abspath('outputdir')
        self.make_dir(self.outputdir)
        # Now update paths based on that, if needed
        self.update_path('fbestp')
        self.update_path('flog')
        self.update_path('fsavefile')
        self.update_path('fsavemodel')
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
            # Open the log file
            if self.flog is not None:
                logfile = open(self.flog, 'w')
            else:
                logfile = None
            # Run the MCMC
            self.outp, self.bestp = mc3.mc.mcmc(self.data, 
                                                self.uncert, 
                                    func      = self.func, 
                                    indparams = self.indparams,
                                    parnames  = self.pnames, 
                                    params    = self.pinit, 
                                    pmin      = self.pmin, 
                                    pmax      = self.pmax, 
                                    stepsize  = self.pstep,
                                    numit     = self.niter, 
                                    burnin    = self.burnin, 
                                    thinning  = self.thinning, 
                                    nchains   = self.nchains, 
                                    walk      = self.alg, 
                                    hsize     = self.hsize, 
                                    plots     = False, 
                                    leastsq   = False, 
                                    log       = logfile, 
                                    savefile  = self.fsavefile,
                                    savemodel = self.fsavemodel)
            # Save bestfit params
            if self.fbestp is not None:
                np.save(self.fbestp, self.bestp)
            # Close the log
            if self.flog is not None:
                logfile.close()
            return self.outp, self.bestp
        else:
            if self.verb:
                print("Sampler is not fully prepared to run. " + \
                      "Correct the above errors and try again.")


def driver(params):
    """
    DEMC with snooker updating algorithm of ter Braak & Vrugt (2008)

    Inputs
    ------
    params: dict. Dictionary of input parameters.  Must include an entry for 
                  burnin, data, flog, fsavefile, fsavemodel, func, hsize, 
                  indparams, nchains, niter, outputdir, pinit, pmax, pmin, 
                  pnames, pstep, savefile, thinning, and uncert.
                  See user manual for descriptions.

    Outputs
    -------
    outp : array. Each set of accepted parameter values.
    bestp: array. Best parameter values.
    """
    if params['flog'] is not None:
        logfile = open(params['flog'], 'w')
    else:
        logfile = None

    # Run the MCMC
    outp, bestp = mc3.mc.mcmc(params['data'], params['uncert'], 
                              func      = params['func'], 
                              indparams = params['indparams'],
                              parnames  = params['pnames'], 
                              params    = params['pinit'], 
                              pmin      = params['pmin'], 
                              pmax      = params['pmax'], 
                              stepsize  = params['pstep'],
                              numit     = params['niter'], 
                              burnin    = params['burnin'], 
                              thinning  = params['thinning'], 
                              nchains   = params['nchains'], 
                              walk      = 'snooker', 
                              hsize     = params['hsize'], 
                              plots     = False, 
                              leastsq   = False, 
                              log       = logfile, 
                              savefile  = params['fsavefile'],
                              savemodel = params['fsavemodel'])

    if params['flog'] is not None:
        logfile.close()

    return outp, bestp


