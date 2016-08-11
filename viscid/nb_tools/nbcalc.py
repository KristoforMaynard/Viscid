#!/usr/bin/env python
# pylint: disable=bad-whitespace

from __future__ import division, print_function
import math
import sys

import numpy as np
import numba as nb

import viscid


__all__ = ["nb_interp", "nb_interp_trilin", "nb_interp_nearest",
           "nb_closest_ind", "nb_closest_preceeding_ind",
           "nb_interp_trilin_x", "nb_interp_nearest_x"]


_NUMBA_NOGIL = True
_NUMBA_CACHE = False



def nb_interp_trilin(vfield, seeds, wrap=True):
    """Trilinear interpolate vfield onto seeds using Numba"""
    return nb_interp(vfield, seeds, wrap=wrap, kind="trilinear")

def nb_interp_nearest(vfield, seeds, wrap=True):
    """Nearest neighbor interpolate vfield onto seeds using Numba"""
    return nb_interp(vfield, seeds, wrap=wrap, kind="nearest")

def nb_interp(vfield, seeds, kind="nearest", wrap=True):
    """Interpolate vfield onto seeds using Numba"""
    kind = kind.strip().lower()

    nb_fld = viscid.make_nb_field(vfield)

    seed_center = seeds.center if hasattr(seeds, 'center') else vfield.center
    if seed_center.lower() in ('face', 'edge'):
        seed_center = 'cell'

    nr_points = seeds.get_nr_points(center=seed_center)
    nr_comps = vfield.nr_comps
    if nr_comps == 0:
        scalar = True
        nr_comps = 1
    else:
        scalar = False

    result = np.empty((nr_points, nr_comps), dtype=nb_fld.data.dtype)

    seeds = viscid.to_seeds(seeds)

    pts = seeds.points(center=seed_center).T.astype(nb_fld.crds.dtype)
    pts = np.ascontiguousarray(pts)

    if kind == "nearest":
        _nb_interp_nearest(nb_fld, pts, result)
    elif kind == "trilinear" or kind == "trilin":
        _nb_interp_trilin(nb_fld, pts, result)
    else:
        raise ValueError("kind '{0}' not understood. Use trilinear or nearest"
                         "".format(kind))

    if scalar:
        result = result[:, 0]

    if wrap:
        if scalar:
            result = seeds.wrap_field(result, name=vfield.name)
        else:
            result = seeds.wrap_field(result, name=vfield.name,
                                      fldtype="vector", layout="interlaced")

    return result


@nb.jit(nopython=True, nogil=_NUMBA_NOGIL, cache=_NUMBA_CACHE)
def _nb_interp_trilin(nb_fld, points, result):
    cached_idx3 = np.zeros((3,), dtype=np.int_)

    for i in range(points.shape[0]):
        for m in range(result.shape[1]):
            result[i, m] = nb_interp_trilin_x(nb_fld, m, points[i, :],
                                              cached_idx3)

@nb.jit(nopython=True, nogil=_NUMBA_NOGIL, cache=_NUMBA_CACHE)
def nb_interp_trilin_x(nb_fld, m, x, cached_idx3):
    """Numba trilinear interpolate at a single point"""
    ix = nb_fld.temp_int3_0
    p = nb_fld.temp_int3_1
    xd = nb_fld.temp_float3_0

    # find closest inds
    for d in range(3):
        if nb_fld.n[d] == 1 or x[d] <= nb_fld.xl[m, d]:
            ind = 0
            ix[d] = ind
            p[d] = 0
            xd[d] = 0.0
        elif x[d] >= nb_fld.xh[m, d]:
            # switch to nearest neighbor for points beyond last value
            ind = nb_fld.n[d] - 1
            p[d] = 0
            xd[d] = 0.0
        else:
            ind = nb_closest_preceeding_ind(nb_fld, m, d, x[d], cached_idx3)
            p[d] = 1
            xd[d] = ((x[d] - nb_fld.crds[m, d, ind]) /
                     (nb_fld.crds[m, d, ind + 1] - nb_fld.crds[m, d, ind]))
        ix[d] = ind

    # INTERLACED ... x first
    c00 = (nb_fld.data[ix[0], ix[1]       , ix[2]       , m] +
           xd[0] * (nb_fld.data[ix[0] + p[0], ix[1]       , ix[2]       , m] -
                    nb_fld.data[ix[0]       , ix[1]       , ix[2]       , m]))
    c10 = (nb_fld.data[ix[0], ix[1] + p[1], ix[2]       , m] +
           xd[0] * (nb_fld.data[ix[0] + p[0], ix[1] + p[1], ix[2]       , m] -
                    nb_fld.data[ix[0]       , ix[1] + p[1], ix[2]       , m]))
    c01 = (nb_fld.data[ix[0], ix[1]       , ix[2] + p[2], m] +
           xd[0] * (nb_fld.data[ix[0] + p[0], ix[1]       , ix[2] + p[2], m] -
                    nb_fld.data[ix[0]       , ix[1]       , ix[2] + p[2], m]))
    c11 = (nb_fld.data[ix[0], ix[1] + p[1], ix[2] + p[2], m] +
           xd[0] * (nb_fld.data[ix[0] + p[0], ix[1] + p[1], ix[2] + p[2], m] -
                    nb_fld.data[ix[0]       , ix[1] + p[1], ix[2] + p[2], m]))
    c0 = c00 + xd[1] * (c10 - c00)
    c1 = c01 + xd[1] * (c11 - c01)
    c = c0 + xd[2] * (c1 - c0)

    return c


@nb.jit(nopython=True, nogil=_NUMBA_NOGIL, cache=_NUMBA_CACHE)
def _nb_interp_nearest(nb_fld, points, result):
    cached_idx3 = np.zeros((3,), dtype=np.int_)

    for i in range(points.shape[0]):
        for m in range(result.shape[1]):
            result[i, m] = nb_interp_nearest_x(nb_fld, m, points[i, :],
                                               cached_idx3)

@nb.jit(nopython=True, nogil=_NUMBA_NOGIL, cache=_NUMBA_CACHE)
def nb_interp_nearest_x(nb_fld, m, x, cached_idx3):
    """Numba nearest neighbor interpolate at a single point"""
    ind0 = nb_closest_ind(nb_fld, m, 0, x[0], cached_idx3)
    ind1 = nb_closest_ind(nb_fld, m, 1, x[1], cached_idx3)
    ind2 = nb_closest_ind(nb_fld, m, 2, x[2], cached_idx3)
    return nb_fld.data[ind0, ind1, ind2, m]


@nb.jit(nopython=True, nogil=_NUMBA_NOGIL, cache=_NUMBA_CACHE)
def nb_closest_preceeding_ind(nb_fld, m, d, value, cached_idx3):
    """Find closest preceeding index of nb_fld.crds[m, d, ?] to value"""
    n = nb_fld.n[d]
    startidx = cached_idx3[d]

    if n == 1:
        ind = 0
    elif nb_fld.uniform_crds:
        frac = (value - nb_fld.xl[m, d]) / (nb_fld.L[m, d])
        i = int(math.floor(nb_fld.nm1[d]) * frac)
        ind = min(max(i, 0), nb_fld.nm2[d])
    else:
        found_ind = 0
        if nb_fld.crds[m, d, startidx] <= value:
            i = startidx
            for i in range(startidx, n - 1):
                if nb_fld.crds[m, d, i + 1] > value:
                    found_ind = 1
                    break
            if not found_ind:
                i = n - 1
        else:
            i = startidx - 1
            for i in range(startidx - 1, -1, -1):
                if nb_fld.crds[m, d, i] <= value:
                    found_ind = 1
                    break
            if not found_ind:
                i = 0
        ind = i

    cached_idx3[d] = ind
    return ind

@nb.jit(nopython=True, nogil=_NUMBA_NOGIL, cache=_NUMBA_CACHE)
def nb_closest_ind(nb_fld, m, d, value, cached_idx3):
    """Find closest index of nb_fld.crds[m, d, ?] to value"""
    preceeding_ind = nb_closest_preceeding_ind(nb_fld, m, d, value, cached_idx3)
    if preceeding_ind == nb_fld.n[d] - 1:
        return nb_fld.n[d] - 1
    else:
        d1 = abs(nb_fld.crds[m, d, preceeding_ind] - value)
        d2 = abs(nb_fld.crds[m, d, preceeding_ind + 1] - value)
        if d1 <= d2:
            return preceeding_ind
        else:
            return preceeding_ind + 1


def _main():
    import os
    # os.environ['NUMBA_NUM_THREADS'] = "2"
    # os.environ['NUMBA_WARNINGS'] = "1"
    # os.environ['NUMBA_DISABLE_JIT'] = "1"
    # os.environ['NUMBA_DUMP_ANNOTATION'] = "1"
    # os.environ['NUMBA_DUMP_IR'] = "1"
    # os.environ['NUMBA_DUMP_BYTECODE'] = "1"
    # os.environ['NUMBA_DUMP_OPTIMIZED'] = "1"
    # os.environ['NUMBA_DUMP_ASSEMBLY'] = "1"

    dt = 'f4'
    n = (128, 64, 64)
    xl = (-30, -30, -30)
    xh = (+30, +30, +30)

    x, y, z = [np.linspace(_l, _h, _n) for _l, _h, _n in zip(xl, xh, n)]
    # x[-1] += 0.1
    f0 = viscid.empty((x, y, z), dtype=dt, center='node')
    f0.data = np.arange(np.prod(f0.shape)).astype(f0.dtype)

    seeds = viscid.Volume(xl=xl, xh=xh, n=n)

    retNB = nb_interp_nearest(f0, seeds)
    assert np.all(f0.data == retNB.data)

    retCY = viscid.interp_trilin(f0, seeds)
    retNB = nb_interp_trilin(f0, seeds)
    assert np.all(retCY.data == retNB.data)

    return 0

if __name__ == "__main__":
    sys.exit(_main())

##
## EOF
##
