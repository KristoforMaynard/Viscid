"""Very hack-and-slash attempt to see if Numba is worth using

Warning:
    The Viscid-Numba API is by no means stable.
"""

from viscid import import_injector

NUMBA_CACHE = True
NUMBA_NOGIL = False

__all__ = ["NUMBA_CACHE", "NUMBA_NOGIL", "nbcalc", "nbfield", "nbintegrate",
           "nbstreamline"]

from viscid.nb_tools import nbcalc
from viscid.nb_tools import nbfield
from viscid.nb_tools import nbintegrate
from viscid.nb_tools import nbstreamline
