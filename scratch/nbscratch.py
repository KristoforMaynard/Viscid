#!/usr/bin/env python

from __future__ import division, print_function
import os
import sys

import numpy as np
import numba as nb

os.environ['NUMBA_DUMP_ANNOTATION'] = "1"

@nb.jit(nopython=True)
def f1(val, ff):
    print(val[0])
    ff(val, inc=np.array(4, dtype=nb.int_))
    print(val[0])
    ff(val, inc=np.array(5, dtype=nb.float_))
    print(val[0])
    ff(val)
    print(val[0])


@nb.jit(nopython=True)
def f2A(val, inc=2):
    val[0] += inc

@nb.jit(nopython=True)
def f2B(val, inc=2):
    val[0] -= inc

def main():
    f1(np.array([1], dtype='i4'), f2A)
    f1(np.array([100.0], dtype='f4'), f2B)
    return 0

if __name__ == "__main__":
    sys.exit(main())

##
## EOF
##
