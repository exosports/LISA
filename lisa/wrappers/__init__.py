__all__ = ['demc_wrapper', 'dnest4_wrapper', 'dream_wrapper', 'dynesty_wrapper', 
           'helper', 'multinest_wrapper', 'polychord_wrapper', 
           'snooker_wrapper', 'ultranest_wrapper']

import warnings

from . import helper

try:
    from . import demc_wrapper
except:
    warnings.warn("DEMC wrapper unable to be imported, possible because it is installed.")
    demc_wrapper = None

try:
    from . import dnest4_wrapper
except:
    warnings.warn("DNest4 wrapper unable to be imported, possibly because it is not installed.")
    dnest4_wrapper = None

try:
    from . import dream_wrapper
except:
    warnings.warn("DREAM wrapper unable to be imported, possibly because it is not installed.")
    dream_wrapper = None

try:
    from . import dynesty_wrapper
except:
    warnings.warn("Dynesty wrapper unable to be imported, possibly because it is not installed.")
    dynesty_wrapper = None

try:
    from . import multinest_wrapper
except:
    warnings.warn("Multinest wrapper unable to be imported, possibly because it is not installed.")
    multinest_wrapper = None

try:
    from . import polychord_wrapper
except:
    warnings.warn("PolyChord wrapper unable to be imported, possibly because it is not installed.")
    polychord_wrapper = None

try:
    from . import snooker_wrapper
except:
    warnings.warn("DEMCzs (snooker) wrapper unable to be imported, possibly because it is not installed.")
    snooker_wrapper = None

try:
    from . import ultranest_wrapper
except:
    warnings.warn("Ultranest wrapper unable to be imported, possibly because it is not installed.")
    ultranest_wrapper = None


