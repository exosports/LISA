import sys, os
import six
import numpy as np

mcpdir = os.path.join(os.path.dirname(__file__), '..', 'modules', 'MCcubed', 
                                                       'MCcubed', 'plots')
sys.path.append(mcpdir)
sys.path.append(os.path.join(mcpdir, '..', 'lib')) # so that mcplots finds binarray
import mcplots as mcp


class BaseSampler(object):
    """
    Parent class for samplers in LISA.  

    Contains the helpinfo attribute, which is a dictionary containing 
    descriptions of each parameter.  The help(param) method returns the 
    dictionary entry for `param`.

    Contains helper methods common to samplers: make_dir, check_none, 
    check_nonnegfloat, check_nonnegint, check_pnames, check_posint, 
    make_abspath, prep_arr, and update_path.  These are used when checking 
    that the user has supplied proper inputs before attempting to run the 
    sampler.
    """
    def __init__(self):
        # Default values
        self.nchains  = 1 # some samplers do not use these params, 
        self.thinning = 1 # but they are required for posterior plots
        self.burnin   = 0 # or when re-loading output posterior
        # Dictionary of parameters and their descriptions
        self.helpinfo = {
        'beta' : 'float. DNest 4 only. From their docs: strength of effect ' + \
                        'to force histogram to equal push.  Default: 100.0', 
        'bound' : 'str. Dynesty only. Option to bound the target ' + \
                       'distribution. Choices: none (sample from unit ' + \
                       'cube), single (one ellipsoid), multi (multiple ' + \
                       'possibly overlapping ellipsoids), balls ' + \
                       '(overlapping balls centered on each live point), ' + \
                       'cubes (overlapping cubes centered on each live ' + \
                       'point).  Default: multi', 
        'burnin' : 'int. Number of initial iterations to be discarded.', 
        'data' : 'array, Numpy binary. Measured data for inference. ' + \
                        'Must be Numpy array, list, or a path to a NPY file.', 
        'dlogz' : 'float. Target evidence uncertainty (stops when below ' + \
                         'this value).  Default: 0.1', 
        'dumper' : 'object.  Polychord only.  Function to output info ' + \
                            'during the inference.  Default: None', 
        'fbestp' : 'str. Filename for array of best-fit parameters. ' + \
                   'Must be NPY file.', 
        'fcheckpoint' : 'str. Dynesty only.  Path to save checkpoint file ' + \
                        'to allow for resuming the run.  If a relative path, ' + \
                        'assumes it is relative to `outputdir`.', 
        'fext' : 'str. File extension for saved plots. ' + \
                      'Options: .png, .pdf  Default: .png', 
        'flog' : 'str. DEMC and snooker only. /path/to/log file to save out. ' + \
                      'Default: MCMC.log, located in `outputdir`', 
        'fprefix' : 'str. Prefix for output filenames. Recommended to be ' + \
                         'a directory (possibly with a prefix for all ' + \
                         'produced files), as this will create a ' + \
                         'subdirectory within `outputdir`. ' + \
                         'Default: pmn/ for multinest, ' + \
                         '         run1 for polychord.', 
        'fsavefile' : 'str. Filename to store parameters explored. ' + \
                            'If relative path, it is considered with ' + \
                            'respect to `outputdir`.  ' + \
                            'Default: `outputdir`/output.npy', 
        'fsavemodel' : 'str. Filename to store models, corresponding to ' + \
                            'the parameters. If relative path, it is ' + \
                            'considered with respect to `outputdir`.  ' + \
                            'If None, file is not saved.  ' + \
                            'Beware: the file may be extremely large ' + \
                            'if the model output has high dimensionality.' + \
                            'Default: None', 
        'frac_remain' : 'float. UltraNest only. Sets the fraction ' + \
                               'remainder when integrating the posterior.', 
        'hsize' : 'int. Snooker only.  Number of samples per chain to seed ' + \
                       'the phase space.  Default: nchains+1', 
        'indparams' : 'list. Additional parameters needed by `func`.', 
        'kll' : 'object.  Datasketches KLL object, for model quantiles. ' + \
                         'Use None if not desired or if Datasketches is ' + \
                         'not installed.  Default: None', 
        'lam' : 'float. DNest 4 only. From their docs: backtracking scale ' + \
                       'length.  Default: 5.0', 
        'Lepsilon' : 'float. UltraNest only. From their docs: "Terminate ' + \
                            'when live point likelihoods are all the same, ' + \
                            'within Lepsilon tolerance."', 
        'loglike' : 'object. Function defining the log likelihood.', 
        'min_ess' : 'int. Minimum effective sample size (ESS). Default: 500', 
        'model' : 'object. Function defining the forward model.', 
        'modelper' : 'int. Sets how to split `fsavemodel` into subfiles.  ' + \
                          'If 0, does not split.  If >0, saves out every ' + \
                          '`modelper` iterations.  E.g., if nchains=10 and ' + \
                          'modelper=5, splits every 50 model evaluations.  ' + \
                          'Default: 0', 
        'multitry' : 'int. DREAM only. Determines whether to use multi-try ' + \
                          'sampling. Default: 5', 
        'nchains' : 'int. Number of parallel samplers. Default: 1', 
        'niter' : 'int. Maximum number of iterations.  Nested samplers ' + \
                       'default to no limit.', 
        'nlive' : 'int. (Minimum) number of live points to use. Default: 500', 
        'nlive_batch': 'int. Dynesty only. From their docs: "The number of ' + \
                            'live points used when adding additional ' + \
                            'samples from a nested sampling run within ' + \
                            'each batch."  Default: 500.', 
        'nlevel' : 'int. DNest4 only. From their docs: Maximum number of ' + \
                        'levels to create.  Default: 30', 
        'nlevelint' : 'int. DNest4 only. Number of moves before creating ' + \
                           'new level. Default: 10000', 
        'nperstep' : 'int. DNest4 only. Number of moves per MCMC ' + \
                          'iteration.  Default: 10000', 
        'nrepeat' : "int. Polychord only.  From their docs: The number of " + \
                         "slice slice-sampling steps to generate a new " + \
                         "point.  Increasing num_repeats increases the " + \
                         "reliability of the algorithm.  Default: None " + \
                         "(uses polychord's default of 5*ndims)", 
        'outputdir' : 'str. path/to/directory where output will be saved.', 
        'perturb' : 'object. DNest4 only. Function that proposes changes ' + \
                            'to parameter values.',  
        'pinit' : 'array, Numpy binary. For MCMCs, initial parameters for ' + \
                            'samplers.  For nested sampling algorithms, ' + \
                            'values are used for parameters that are held ' + \
                            'constant, if any.' + \
                        'Must be Numpy array, list, or a path to a NPY file.',
        'pmax' : 'array, Numpy binary. Maximum value for each parameter.' + \
                        'Must be Numpy array, list, or a path to a NPY file.',
        'pmin' : 'array, Numpy binary. Minimum value for each parameter.' + \
                        'Must be Numpy array, list, or a path to a NPY file.',
        'pnames' : 'array. Name of each parameter (can use some LaTeX if ' + \
                          'desired).' + \
                          'Must be Numpy array or list.', 
        'prior' : 'object. Function defining the prior.', 
        'pstep' : 'array. Step size for each parameter.  For MCMCs, only ' + \
                         'matters for the initial samples and determining ' + \
                         'constant parameters, as step size is ' + \
                         'automatically adjusted.  For nested sampling ' + \
                         'algorithms, only used to determine constant ' + \
                         'parameters.', 
        'resample' : 'float. DNest4 only. Must be non-negative.  ' + \
                         'If >0, corresponds to a factor affecting the ' + \
                         'number of draws from the posterior.  If 0, no ' + \
                         'resampling is performed.  Default: 100', 
        'resume' : 'bool. Determines whether to resume a previous run, if ' + \
                         'possible. Default: False', 
        'sample' : 'str. Dynesty only. Sampling method. Choices ' + \
                        '(descriptions from their docs): unif (uniform ' + \
                        'sampling), rwalk (random walks from current live ' + \
                        'point), rstagger (random "staggering" away from ' + \
                        'current live point), slice (multivariate slice ' + \
                        'sampling), rslice (random slice sampling), hslice ' + \
                        '("Hamiltonian" slice sampling), auto ' + \
                        '(automatically selected based on problem ' + \
                        'dimensionality).  Default: auto', 
        'thinning' : 'int. Thinning factor for the posterior ' + \
                          '(keep every N iterations). ' + \
                          'Example: a thinning factor of 3 will keep every ' + \
                          'third iteration.  Only recommended when the ' + \
                          'computed posterior is extremely large.  ' + \
                          'Default: 1', 
        'truepars' : 'array, Numpy binary. Known true values for the model ' + \
                          'parameters.  If unknown, use None.  Default: None', 
        'uncert' : 'array, Numpy binary. Data uncertainties.' + \
                        'Must be Numpy array, list, or a path to a NPY file.', 
        'verb' : 'int. Verbosity level.  If 0, only essential messages are ' + \
                      'printed by LISA.  If 1, prints additional messages ' + \
                      '(e.g., loading data files).  Default: 0'
        }

    def help(self, par):
        """
        Returns information about a parameter

        Inputs
        ------
        par: string. Parameter of interest.
        """
        if par in self.helpinfo.keys():
            print(par, ":", self.helpinfo[par])
        else:
            print(par, "is not a parameter for this sampler.")

    def make_dir(self, some_dir):
        """
        Handles creation of a directory.

        Inputs
        ------
        some_dir: string. Directory to be created.

        Outputs
        -------
        None. Creates `some_dir` if it does not already exist. 
        Raises an error if the directory cannt be created.
        """
        try:
          os.mkdir(some_dir)
        except OSError as e:
          if e.errno == 17: # Already exists
            pass
          else:
            print("Cannot create folder '{:s}'. {:s}.".format(some_dir,
                                                  os.strerror(e.errno)))
            sys.exit()
        return

    def check_none(self, attr):
        """
        Checks attributes that must be specified.
        """
        if getattr(self, attr) is None:
            print(attr, "must be specified.")
            self.unprepared += 1

    def check_nonnegfloat(self, attr):
        """
        Checks attributes that must be non-negative floats.
        """
        if type(getattr(self, attr)) == int:
            setattr(self, attr, float(getattr(self, attr)))
        if getattr(self, attr) is None or getattr(self, attr) < 0 or \
           type(getattr(self, attr)) != float:
            print(attr, "must be a non-negative float.")
            self.unprepared += 1

    def check_nonnegint(self, attr):
        """
        Checks attributes that must be non-negative integers.
        """
        val = getattr(self, attr)
        if val is None or val < 0 or \
           type(val) != int:
            print(attr, "must be a non-negative integer.  Given:", val)
            self.unprepared += 1

    def check_pnames(self):
        """
        Creates an array of parameter names if the user does not provide them
        ** MODIFIES THE OBJECT'S pnames ATTRIBUTE IF IT IS None **
        """
        # Set default parameter names:
        if self.pnames is None and self.pstep is not None:
            if self.verb:
                print("Using default parameter names.")
            npars       = self.pstep.size
            namelen     = int(2+np.log10(np.amax([npars-1,1])))
            self.pnames = np.zeros(npars, "|S%d"%namelen if six.PY2 
                                     else "<U%d"%namelen)
            for i in np.arange(npars):
                self.pnames[i] = "P" + str(i).zfill(namelen-1)
        elif type(self.pnames) == list:
            if self.verb:
                print("Converting pnames list into Numpy array")
            self.pnames = np.asarray(self.pnames)

    def check_posint(self, attr):
        """
        Checks attributes that must be a positive integer.
        """
        val = getattr(self, attr)
        if val is None or val < 1 or type(val) != int:
            print(attr, "must be a positive integer.  Given:", val)
            self.unprepared += 1

    def make_abspath(self, attr):
        """
        Ensures a path is an absolute path and that it exists.  Returns True on 
        success or False on failure.
        """
        if getattr(self, attr) is not None:
            # Ensure absolute path
            if not os.path.isabs(getattr(self, attr)):
                setattr(self, attr, os.path.abspath(getattr(self, attr)))
            # Ensure directory exists
            if not os.path.exists(getattr(self, attr)):
                self.make_dir(getattr(self, attr))
            return True
        else:
            print(attr, "must be specified.")
            self.unprepared += 1
            return False

    def prep_arr(self, attr):
        """
        Prepares attributes that may be an array.
        ** MODIFIES THE OBJECT'S ATTRIBUTE IF IT IS NOT None **
        """
        if getattr(self, attr) is None:
            print(attr, "must be specified as either a Numpy array or " + \
                  "a path to a Numpy binary file (.npy).")
            self.unprepared += 1
        elif type(getattr(self, attr)) == list:
            if self.verb:
                print("Converting", attr, "list into Numpy array")
            setattr(self, attr, np.asarray(getattr(self, attr)))
        elif type(getattr(self, attr)) == str:
            if self.verb:
                print("Loading data Numpy binary file")
            setattr(self, attr, np.load(getattr(self, attr)))

    def update_path(self, attr):
        """
        Sets a path to an absolute path, if it is not already.
        ** MODIFIES THE OBJECT'S ATTRIBUTE IF IT IS NOT AN ABSOLUTE PATH **
        """
        if getattr(self, attr) is not None:
            if not os.path.isabs(getattr(self, attr)):
                setattr(self, attr, os.path.join(getattr(self, 'outputdir'), 
                                                 getattr(self, attr)))

    def make_plots(self):
        """
        Produces posterior plots
        """
        if hasattr(self, 'outp') or os.path.exists(self.fsavefile):
            if not hasattr(self, 'outp'):
                self.outp = np.load(self.fsavefile)
            mcp.trace(self.outp, parname=self.pnames[self.pstep>0], 
                      thinning=self.thinning, 
                      sep=np.size(self.outp[0]//self.nchains), 
                      savefile=os.path.join(self.outputdir, "trace"+self.fext),
                      truepars=self.truepars)
            mcp.histogram(self.outp, parname=self.pnames[self.pstep>0], 
                          thinning=self.thinning, 
                          savefile=os.path.join(self.outputdir, 
                                                "posterior"+self.fext),
                          truepars=self.truepars, density=True)
            mcp.pairwise(self.outp, parname=self.pnames[self.pstep>0], 
                         thinning=self.thinning, 
                         savefile=os.path.join(self.outputdir, 
                                               "pairwise"+self.fext),
                         truepars=self.truepars)
        else:
            print("Attempted to produce posterior plots, but the " + \
                  "inference has not yet successfully executed.")
            print("Execute the run() method and try again.")


