#!/usr/bin/env python
# pylint: disable=bad-whitespace,unused-argument
"""Integrators that use Viscid Fields wrapped in Numba

Warning:
    The Viscid-Numba API is by no means stable.
"""

from __future__ import division, print_function
import math
import sys

import numpy as np
import numba as nb

# import viscid
from viscid.nb_tools.nbcalc import nb_interp_trilin_x


__all__ = ["nb_euler1", "nb_euler1a"]


_NUMBA_NOGIL = True
_NUMBA_CACHE = False


@nb.njit(fastmath=False, nogil=_NUMBA_NOGIL, cache=_NUMBA_CACHE)
def nb_euler1(nb_fld, x, ds, tol_lo, tol_hi, fac_refine, fac_coarsen,
              smallest_step, largest_step, vscale, cached_idx3):
    """1st order Euler integration step"""
    v0 = vscale[0] * nb_interp_trilin_x(nb_fld, 0, x, cached_idx3)
    v1 = vscale[1] * nb_interp_trilin_x(nb_fld, 1, x, cached_idx3)
    v2 = vscale[2] * nb_interp_trilin_x(nb_fld, 2, x, cached_idx3)
    vmag = math.sqrt(v0**2 + v1**2 + v2**2)
    if vmag == 0.0 or math.isnan(vmag):
        # logger.warn("vmag issue at: {0} {1} {2}, [{3}, {4}, {5}] == |{6}|".format(
        #                 x[0], x[1], x[2], vx, vy, vz, vmag))
        return 1
    x[0] += ds[0] * v0 / vmag
    x[1] += ds[0] * v1 / vmag
    x[2] += ds[0] * v2 / vmag
    return 0

@nb.njit(fastmath=False, nogil=_NUMBA_NOGIL, cache=_NUMBA_CACHE)
def nb_euler1a(nb_fld, x, ds, tol_lo, tol_hi, fac_refine, fac_coarsen,
               smallest_step, largest_step, vscale, cached_idx3):
    """Adaptive 1st order Euler integration step

    works by going backward to see how close we get to our starting
    point
    """
    xA = nb_fld.temp_float3_1
    xB = nb_fld.temp_float3_2

    while True:
        # go forward
        v0 = vscale[0] * nb_interp_trilin_x(nb_fld, 0, x, cached_idx3)
        v1 = vscale[1] * nb_interp_trilin_x(nb_fld, 1, x, cached_idx3)
        v2 = vscale[2] * nb_interp_trilin_x(nb_fld, 2, x, cached_idx3)
        vmag = math.sqrt(v0**2 + v1**2 + v2**2)
        if vmag == 0.0 or math.isnan(vmag):
            # logger.warn("vmag issue at: {0} {1} {2}, [{3}, {4}, {5}] == |{6}|".format(
            #                 x[0], x[1], x[2], vx, vy, vz, vmag))
            return 1
        xA[0] = x[0] + ds[0] * v0 / vmag
        xA[1] = x[1] + ds[0] * v1 / vmag
        xA[2] = x[2] + ds[0] * v2 / vmag

        # now go backward
        v0 = vscale[0] * nb_interp_trilin_x(nb_fld, 0, xA, cached_idx3)
        v1 = vscale[1] * nb_interp_trilin_x(nb_fld, 1, xA, cached_idx3)
        v2 = vscale[2] * nb_interp_trilin_x(nb_fld, 2, xA, cached_idx3)
        vmag = math.sqrt(v0**2 + v1**2 + v2**2)
        if vmag == 0.0 or math.isnan(vmag):
            # logger.warn("vmag issue at: {0} {1} {2}, [{3}, {4}, {5}] == |{6}|".format(
            #                 x[0], x[1], x[2], vx, vy, vz, vmag))
            return 1
        xB[0] = xB[0] - ds[0] * v0 / vmag
        xB[1] = xB[1] - ds[0] * v1 / vmag
        xB[2] = xB[2] - ds[0] * v2 / vmag

        # now maybe update ds?
        dist_sq = ((x[0] - xB[0])**2 + (x[1] - xB[1])**2 + (x[2] - xB[2])**2)


        if dist_sq > (tol_hi * ds[0])**2:
            # logger.debug("Refining ds: {0} -> {1}".format(
            #     deref(ds), fac_refine * deref(ds)))
            if ds[0] <= smallest_step:
                break
            else:
                ds[0] = max(fac_refine * ds[0], 1.0 * smallest_step)
                continue
        elif dist_sq < (tol_lo * ds[0])**2:
            # logger.debug("Coarsening ds: {0} -> {1}".format(
            #     deref(ds), fac_coarsen * deref(ds)))
            ds[0] = min(fac_coarsen * ds[0], 1.0 * largest_step)
            break
        else:
            break

    x[0] = xA[0]
    x[1] = xA[1]
    x[2] = xA[2]
    return 0

def _main():
    return 0

if __name__ == "__main__":
    sys.exit(_main())

##
## EOF
##
