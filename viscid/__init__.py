# -*- coding: utf-8 -*-
"""A set of python modules that aid in plotting scientific data

Plotting depends on matplotlib and/or mayavi and file reading uses h5py
and to read hdf5 / xdmf files.

Note:
    Modules in calculator and plot must be imported explicitly since
    they have side effects on import.

Attributes:
    logger (logging.Logger): a logging object whose verbosity can be
        set from the command line using
        :py:func`viscid.vutil.common_argparse`.
"""

from __future__ import print_function
import logging
import os
import re
import signal
import sys
import textwrap

import numpy

from viscid import _rc
from viscid.compat.vimportlib import import_module


__version__ = """0.99.6.dev0"""

__all__ = ['amr_field',
           'amr_grid',
           'bucket',
           'coordinate',
           'cotr',
           'dataset',
           'dipole',
           'extools',
           'field',
           'fluidtrace',
           'grid',
           'mapfield',
           'multiplot',
           'npdatetime',
           'parallel',
           'pyeval',
           'seed',
           'sliceutil',
           'tree',
           'verror',
           'vjson',
           'vutil',
           'calculator',  # packages
           'compat',
           'cython',
           'plot',
           'readers',
          ]


#########################################
# setup logger for use throughout viscid
logger = logging.getLogger("viscid")
_handler = logging.StreamHandler()
_handler.setFormatter(logging.Formatter(fmt="%(levelname)s: %(message)s"))
logger.addHandler(_handler)


class _CustomFilter(logging.Filter, object):
    def filter(self, record):
        if '\n' not in record.msg:
            record.msg = '\n'.join(textwrap.wrap(record.msg, width=65))
        spaces = ' ' * (len(record.levelname) + 2)
        record.msg = record.msg.replace('\n', '\n' + spaces)
        return super(_CustomFilter, self).filter(record)


logger.addFilter(_CustomFilter())
logger.propagate = False
del _handler


###################################################################
# this is thunder-hacky, but it's a really simple way to import
# everything in __all__ and also, if those module have an __all__,
# then bring that stuff into this namespace too
def _on_injected_import_error(name, exception, quiet=False):
    if not quiet:
        logger.error(str(exception))
        logger.error("Viscid tried to import {0}, but the import failed.\n"
                     "This module will not be available".format(name))

def import_injector(attr_list, namespace, package=None, quiet=False,
                    fatal=False):
    """import list of modules and consume their __all__ attrs"""
    additional = []
    for s in list(attr_list):
        try:
            m = import_module("." + s, package=package)
            namespace[s] = m
            # print(">", package, ">", s)
            # print(">", package, ">", s, "::", getattr(m, "__all__", None))
            if hasattr(m, "__all__"):
                all_subattrs = getattr(m, "__all__")
                additional += all_subattrs
                for sub in all_subattrs:
                    # print("    ", sub, "=", getattr(m, sub))
                    namespace[sub] = getattr(m, sub)
        except ImportError as e:
            if s not in namespace:
                _on_injected_import_error(s, e, quiet=quiet)
                attr_list.remove(s)
                if fatal:
                    raise
    attr_list += additional

import_injector(__all__, globals(), package="viscid")


##############################################################
# now add some other random things into the __all__ namespace
__all__.append("logger")

# set the sample_dir so that it always points to something useful
# - for installed distribution
sample_dir = os.path.join(os.path.dirname(__file__), 'sample')
# - for in-place distribution
if not os.path.isdir(sample_dir):
    sample_dir = os.path.join(os.path.dirname(__file__), '..', 'sample')
    sample_dir = os.path.abspath(sample_dir)
# - is there a 3rd option? this shouldn't happen
if not os.path.isdir(sample_dir):
    sample_dir = "SAMPLE-DIR-NOT-FOUND"

__all__.append("sample_dir")

# now this is just too cute to pass up :)
if sys.version_info[0] >= 3:
    # hide setting to a unicode variable name in an exec b/c otherwise
    # this file wouldn't parse in python2x
    exec("π = numpy.pi")  # pylint: disable=exec-used
    __all__ += ["π"]

# apply settings in the rc file
_rc.load_rc_file("~/.viscidrc")

def check_version():
    """Check status of viscid and associated libraries and modules"""
    print("Viscid located at:", __file__)
    print()
    print("Viscid version:", __version__)
    print()
    try:
        import matplotlib
        print("Matplotlib version:", matplotlib.__version__)
    except ImportError:
        print("Matplotlib not installed")
    try:
        import mayavi
        print("Mayavi version:", mayavi.__version__)
        try:
            import vtk
            print("VTK version:", vtk.VTK_VERSION)
        except ImportError:
            print("VTK python module not installed")
    except ImportError:
        print("Mayavi not installed")
    print()

    def print_err(*args, **kwargs):
        kwargs.pop('file', '')
        print(*args, file=sys.stderr, **kwargs)

    try:
        from viscid.readers import _jrrle
        print("Fortran modules are correctly built.")
    except ImportError:
        print_err("WARNING: jrrle reader is not available. If you need this")
        print_err("         functionality, please ensure that you have a working")
        print_err("         fortran compiler and reinstall (or rebulid) Viscid.")
        print_err()
    if isinstance(cyfield, cython._dummy):
        print_err("WARNING: cython modules (interpolation and streamlines) are not")
        print_err("         available. To use these functions, please ensure that you")
        print_err("         have a C compiler compatable with your version of")
        print_err("         Python / Numpy and reinstall (or rebulid) Viscid.")
        print_err()
    else:
        print("Cython modules are correctly built.")

__all__.append("check_version")

if hasattr(signal, 'SIGINFO'):
    # this is useful for debugging, ie, immediately do a pdb.set_trace()
    # on the SIGINFO signal
    def _set_trace(seg, frame):  # pylint: disable=unused-argument
        import pdb
        pdb.set_trace()
    signal.signal(signal.SIGINFO, _set_trace)
    # print("Trigger pdb with SIGINFO (ctrl + T)")
