#!/usr/bin/env python
# pylint: disable=bad-whitespace,unused-argument
"""Calculate streamlines using Numba

Warning:
    The Viscid-Numba API is by no means stable.
"""

from __future__ import division, print_function
import math
import sys

import numpy as np
import numba as nb

import viscid
from viscid.nb_tools import nbintegrate


__all__ = ["nb_calc_streamlines"]


_NUMBA_NOGIL = True
_NUMBA_CACHE = False

# IMPORTANT: These constants MUST not change after numba code is compiled
#            since the jit compiler treats global variables as compile-time
#            constants

# should be 3.4028235e+38 ?
MAX_FLOAT = 1e37
# should be 1.7976931348623157e+308 ?
MAX_DOUBLE = 1e307


EULER1 = 1  # euler1 non-adaptive
RK2 = 2  # rk2 non-adaptive
RK12 = 3  # euler1 + rk2 adaptive
EULER1A = 4  # euler 1st order adaptive
METHOD = {EULER1: EULER1, "euler": EULER1, "euler1": EULER1,
          # RK2: RK2, "rk2": RK2,
          # RK12: RK12, "rk12": RK12,
          EULER1A: EULER1A, "euler1a": EULER1A,
         }

DIR_FORWARD = 1
DIR_BACKWARD = 2
DIR_BOTH = 3  # = DIR_FORWARD | DIR_BACKWARD

OUTPUT_STREAMLINES = 1
OUTPUT_TOPOLOGY = 2
OUTPUT_BOTH = 3  # = OUTPUT_STREAMLINES | OUTPUT_TOPOLOGY

# topology will be 1+ of these flags binary or-ed together
#                                bit #   4 2 0 8 6 4 2 0  Notes         bit
END_NONE = 0                         # 0b000000000000000 not ended yet    X
END_IBOUND = 1                       # 0b000000000000001                  0
END_IBOUND_NORTH = 2 | END_IBOUND    # 0b000000000000011  == 3            1
END_IBOUND_SOUTH = 4 | END_IBOUND    # 0b000000000000101  == 5            2
END_OBOUND = 8                       # 0b000000000001000                  3
END_OBOUND_XL = 16 | END_OBOUND      # 0b000000000011000  == 24           4
END_OBOUND_XH = 32 | END_OBOUND      # 0b000000000101000  == 40           5
END_OBOUND_YL = 64 | END_OBOUND      # 0b000000001001000  == 72           6
END_OBOUND_YH = 128 | END_OBOUND     # 0b000000010001000  == 136          7
END_OBOUND_ZL = 256 | END_OBOUND     # 0b000000100001000  == 264          8
END_OBOUND_ZH = 512 | END_OBOUND     # 0b000001000001000  == 520          9
END_CYCLIC = 1024                    # 0b000010000000000  !!NOT USED!!   10
END_OTHER = 2048                     # 0b000100000000000                 11
END_MAXIT = 4096 | END_OTHER         # 0b001100000000000  == 6144        12
END_MAX_LENGTH = 8192 | END_OTHER    # 0b010100000000000  == 10240       13
END_ZERO_LENGTH = 16384 | END_OTHER  # 0b100100000000000  == 18432       14

# IMPORTANT! If TOPOLOGY_MS_* values change, make sure to also change the
# values in viscid/cython/__init__.py since those are used if the cython
# code is not built

# ok, this is over complicated, but the goal was to or the topology value
# with its neighbors to find a separator line... To this end, or-ing two
# _C_END_* values doesn't help, so before streamlines returns, it will
# replace the numbers that mean closed / open with powers of 2, that way
# we end up with topology as an actual bit mask
TOPOLOGY_MS_NONE = 0  # no translation needed
TOPOLOGY_MS_CLOSED = 1  # translated from 5, 6, 7(4|5|6)
TOPOLOGY_MS_OPEN_NORTH = 2  # translated from 13 (8|5)
TOPOLOGY_MS_OPEN_SOUTH = 4  # translated from 14 (8|6)
TOPOLOGY_MS_SW = 8  # no translation needed
# TOPOLOGY_MS_CYCLIC = 16  # no translation needed

TOPOLOGY_MS_SEPARATOR = (TOPOLOGY_MS_CLOSED | TOPOLOGY_MS_OPEN_NORTH |
                         TOPOLOGY_MS_OPEN_SOUTH | TOPOLOGY_MS_SW)

_TOPO_STYLE_GENERIC = 0
_TOPO_STYLE_MSPHERE = 1


def nb_calc_streamlines(vfield, seeds, nr_procs=1, force_subprocess=False,
                        threads=True, chunk_factor=1, wrap=True, **kwargs):
    """Calculate streamlines using Numba"""
    if vfield.nr_sdims != 3 or vfield.nr_comps != 3:
        raise ValueError("Streamlines are only written in 3D.")
    nb_fld = viscid.make_nb_field(vfield)

    seed_center = seeds.center if hasattr(seeds, 'center') else vfield.center
    if seed_center.lower() in ('face', 'edge'):
        seed_center = 'cell'

    nr_points = seeds.get_nr_points(center=seed_center)

    # topo = np.empty((nr_points, ), dtype=np.int_)

    seeds = viscid.to_seeds(seeds)

    pts = seeds.points(center=seed_center).T.astype(nb_fld.crds.dtype)
    pts = np.ascontiguousarray(pts)

    # TODO: parallelize
    lines, topology_ndarr = _nb_streamline_shim(nb_fld, pts, **kwargs)
    # TODO: reassemble lines / topology_ndarr after the join

    if wrap:
        topo = seeds.wrap_field(topology_ndarr, name="Topology")
    else:
        topo = topology_ndarr

    return lines, topo

def _nb_streamline_shim(nb_fld, seeds, ds0=0.0, ibound=0.0, obound0=None,
                        obound1=None, stream_dir=DIR_BOTH, output=OUTPUT_BOTH,
                        method=EULER1, maxit=90000, max_length=1e30, tol_lo=1e-3,
                        tol_hi=1e-2, fac_refine=0.5, fac_coarsen=1.25,
                        smallest_step=1e-4, largest_step=1e2, topo_style="msphere"):
    # unify the types of all these kwargs so there isn't any casting in
    # the jitted function
    crd_dtype = nb_fld.crds.dtype

    ds0 = np.array(ds0, dtype=crd_dtype)
    ibound = np.array(ibound, dtype=crd_dtype)
    # stream_dir = np.array(stream_dir, dtype=np.int_)
    # output = np.array(output, dtype=np.int_)
    # method = np.array(method, dtype=np.int_)
    # maxit = np.array(maxit, dtype=np.int_)
    max_length = np.array(max_length, dtype=crd_dtype)
    tol_lo = np.array(tol_lo, dtype=crd_dtype)
    tol_hi = np.array(tol_hi, dtype=crd_dtype)
    fac_refine = np.array(fac_refine, dtype=crd_dtype)
    fac_coarsen = np.array(fac_coarsen, dtype=crd_dtype)
    smallest_step = np.array(smallest_step, dtype=crd_dtype)
    largest_step = np.array(largest_step, dtype=crd_dtype)

    method = METHOD[method]  # yo dawg

    # ---------------------------
    # pre-process outer boundary
    np_obound0 = np.empty((3,), dtype=nb_fld.crds.dtype)
    np_obound1 = np.empty((3,), dtype=nb_fld.crds.dtype)
    vscale = np.empty((3,), dtype=nb_fld.crds.dtype)

    if obound0 is not None:
        for i in range(3):
            np_obound0[i] = obound0[i]
    if obound1 is not None:
        for i in range(3):
            np_obound1[i] = obound1[i]
    for i in range(3):
        if nb_fld.n[i] == 1:
            if obound0 is None:
                np_obound0[i] = -1e30
            if obound1 is None:
                np_obound1[i] = 1e30
            vscale[i] = 0.0
        else:
            if obound0 is None:
                np_obound0[i] = nb_fld.xlnc[i]
            if obound1 is None:
                np_obound1[i] = nb_fld.xhnc[i]
            vscale[i] = 1.0

    if ds0 == 0.0:
        ds0 = np.array(0.5, dtype=crd_dtype) * nb_fld.min_dx

    if isinstance(topo_style, viscid.string_types):
        if topo_style.strip().lower() == "msphere":
            topo_style = _TOPO_STYLE_MSPHERE
        else:
            topo_style = _TOPO_STYLE_GENERIC

    return _nb_streamline(nb_fld, seeds, ds0, ibound, np_obound0, np_obound1,
                          stream_dir, output, method, maxit, max_length, tol_lo,
                          tol_hi, fac_refine, fac_coarsen, smallest_step,
                          largest_step, topo_style, vscale)

# @nb.jit(nopython=False, nogil=False, cache=_NUMBA_CACHE)
def _nb_streamline(nb_fld, seeds, ds0, ibound, obound0, obound1, stream_dir,
                   output, method, maxit, max_length, tol_lo, tol_hi, fac_refine,
                   fac_coarsen, smallest_step, largest_step, topo_style, vscale):
    # ========
    # nr_segs = 0
    maxit2 = 2 * maxit + 1
    nr_streams = seeds.shape[0]
    # --------
    # 2 (0=backward, 1=forward), 3 (x, y, z), maxit points in the line
    line_ndarr = np.empty((3, maxit2), dtype=nb_fld.crds.dtype)
    lines = []
    topology_ndarr = np.empty((nr_streams,), dtype=np.int_)
    # --------
    x0 = np.empty((3,), dtype=nb_fld.crds.dtype)
    s = np.empty((3,), dtype=nb_fld.crds.dtype)
    ds = np.empty((1,), dtype=nb_fld.crds.dtype)
    cached_idx3 = np.zeros((3,), dtype=np.int_)
    line_ends = np.empty((2,), dtype=np.int_)
    _dir_d = np.array([-1, 1], dtype=np.int_)
    # ========

    # # FIXME: allow all integrators
    # integrate_func = nbintegrate.nb_euler1

    if topo_style == _TOPO_STYLE_MSPHERE:
        end_flags_to_topology = end_flags_to_topology_msphere
    else:
        end_flags_to_topology = end_flags_to_topology_generic

    for istream in range(seeds.shape[0]):
        x0[0] = seeds[istream, 0]
        x0[1] = seeds[istream, 1]
        x0[2] = seeds[istream, 2]

        if output & OUTPUT_STREAMLINES:
            line_ndarr[0, maxit] = x0[0]
            line_ndarr[1, maxit] = x0[1]
            line_ndarr[2, maxit] = x0[2]
        line_ends[0] = maxit - 1
        line_ends[1] = maxit + 1

        end_flags = _nb_streamline_single(nb_fld, ds0, ibound, obound0, obound1,
                                          stream_dir, output, method, maxit2, max_length,
                                          tol_lo, tol_hi, fac_refine, fac_coarsen,
                                          smallest_step, largest_step, vscale,
                                          cached_idx3, x0, s, _dir_d, ds,
                                          line_ends, line_ndarr)

        if output & OUTPUT_STREAMLINES:
            line_cat = np.array(line_ndarr[:, line_ends[0] + 1:line_ends[1]])
            lines.append(line_cat)

        if output & OUTPUT_TOPOLOGY:
            topology_ndarr[istream] = end_flags_to_topology(end_flags)

    return lines, topology_ndarr

@nb.jit(nopython=True, nogil=False, cache=_NUMBA_CACHE)
def _nb_streamline_single(nb_fld, ds0, ibound, obound0, obound1, stream_dir,
                          output, method, maxit2, max_length, tol_lo, tol_hi, fac_refine,
                          fac_coarsen, smallest_step, largest_step, vscale,
                          cached_idx3, x0, s, _dir_d, ds, line_ends, line_ndarr):
    end_flags = END_NONE

    for i in range(2):
        d = _dir_d[i]
        # i = 0, d = -1, backward ;; i = 1, d = 1, forward
        if d < 0 and not stream_dir & DIR_BACKWARD:
            continue
        elif d > 0 and not stream_dir & DIR_FORWARD:
            continue

        ds[0] = d * ds0
        stream_length = 0.0

        s[0] = x0[0]
        s[1] = x0[1]
        s[2] = x0[2]

        it = line_ends[i]
        done = END_NONE

        while it >= 0 and it < maxit2:
            # nr_segs += 1
            pre_ds = abs(ds[0])

            # TODO: if amrfld.nr_patches > 1


            # print(">>", istream, i, it)
            if method == EULER1:
                ret = nbintegrate.nb_euler1(nb_fld, s, ds, tol_lo, tol_hi, fac_refine,
                                            fac_coarsen, smallest_step, largest_step,
                                            vscale, cached_idx3)
            elif method == EULER1A:
                ret = nbintegrate.nb_euler1a(nb_fld, s, ds, tol_lo, tol_hi, fac_refine,
                                             fac_coarsen, smallest_step, largest_step,
                                             vscale, cached_idx3)
            else:
                ret = 0
                raise ValueError("Invalid Method")

            if abs(ds[0]) >= pre_ds:
                stream_length += pre_ds
            else:
                stream_length += abs(ds[0])

            # if istream == 0:
            #     print(s[2], s[1], s[0])
            # ret is non 0 when |v_mv| == 0
            if ret != 0:
                done = END_ZERO_LENGTH
                break

            if output & OUTPUT_STREAMLINES:
                line_ndarr[0, it] = s[0]
                line_ndarr[1, it] = s[1]
                line_ndarr[2, it] = s[2]
            it += d

            # end conditions
            done = classify_endpoint(s, stream_length, ibound,
                                     obound0, obound1, max_length, ds, x0)
            if done:
                break
        if done == END_NONE:
            done = END_OTHER | END_MAXIT

        line_ends[i] = it
        end_flags |= done
    return end_flags

@nb.jit(nopython=True, nogil=False, cache=_NUMBA_CACHE)
def classify_endpoint(pt, length, ibound, obound0, obound1, max_length, ds, pt0):
    done = END_NONE
    rsq = pt[0]**2 + pt[1]**2 + pt[2]**2

    if rsq < ibound**2:
        if pt[2] >= 0.0:
            done = END_IBOUND_NORTH
        else:
            done = END_IBOUND_SOUTH
    elif pt[0] < obound0[0]:
        done = END_OBOUND_XL
    elif pt[1] < obound0[1]:
        done = END_OBOUND_YL
    elif pt[2] < obound0[2]:
        done = END_OBOUND_ZL
    elif pt[0] > obound1[0]:
        done = END_OBOUND_XH
    elif pt[1] > obound1[1]:
        done = END_OBOUND_YH
    elif pt[2] > obound1[2]:
        done = END_OBOUND_ZH
    elif length > max_length:
        done = END_MAX_LENGTH

    # if we are within 0.05 * ds of the initial position
    # distsq = (pt0[0] - pt[0])**2 + \
    #          (pt0[1] - pt[1])**2 + \
    #          (pt0[2] - pt[2])**2
    # if distsq < (0.05 * ds[0])**2:
    #     # print("cyclic field line")
    #     done = END_CYCLIC
    #     break

    return done

@nb.jit(nopython=True, nogil=False, cache=_NUMBA_CACHE)
def end_flags_to_topology_msphere(end_flags):
    topo = 0
    mask_open_north = END_IBOUND_NORTH | END_OBOUND
    mask_open_south = END_IBOUND_SOUTH | END_OBOUND

    # order of these if statements matters!
    if end_flags & END_OTHER:
        topo = end_flags
    # elif topo & END_CYCLIC:
    #     return TOPOLOGY_MS_CYCLYC
    elif end_flags & (mask_open_north) == mask_open_north:
        topo = TOPOLOGY_MS_OPEN_NORTH
    elif end_flags & (mask_open_south) == mask_open_south:
        topo = TOPOLOGY_MS_OPEN_SOUTH
    elif end_flags == 3 or end_flags == 5 or end_flags == 7:
        topo = TOPOLOGY_MS_CLOSED
    else:
        topo = TOPOLOGY_MS_SW

    # print("::", type(end_flags), "->", type(topo))

    return topo

@nb.jit(nopython=True, nogil=False, cache=_NUMBA_CACHE)
def end_flags_to_topology_generic(end_flags):
    return end_flags


def _main():
    return 0

if __name__ == "__main__":
    sys.exit(_main())

##
## EOF
##
