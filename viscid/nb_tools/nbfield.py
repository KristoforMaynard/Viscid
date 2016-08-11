#!/usr/bin/env python
# pylint: disable=attribute-defined-outside-init
"""Numba wrappers around Viscid Fields

Warning:
    The Viscid-Numba API is by no means stable.
"""

from __future__ import division, print_function
import sys

import numpy as np
import numba as nb
from numba import jitclass  #  <- turn numba version error into an import error


__all__ = ['make_nb_field', 'NbFldTypes', 'NbFld_F4_Crds_F4',
           'NbFld_F8_Crds_F8', 'NbFld_I4_Crds_F8', 'NbFld_I8_Crds_F8']


class NbFld(object):
    """Numba Field Struct Base Class"""
    def __init__(self):
        pass


nb_fld_common = [("uniform_crds", nb.bool_),
                 ("n", nb.int_[::1]),
                 ("nm1", nb.int_[::1]),
                 ("nm2", nb.int_[::1]),
                 ("nr_nodes", nb.int_[::1]),
                 ("nr_cells", nb.int_[::1]),
                 ("temp_int3_0", nb.int_[::1]),
                 ("temp_int3_1", nb.int_[::1]),
                 ("temp_int3_2", nb.int_[::1]),
                ]
NbFld_F8_Crds_F8 = jitclass(nb_fld_common +
                            [("data", nb.f8[:, :, :, ::1]),
                             ("crds", nb.f8[:, :, ::1]),
                             ("crds_nc", nb.f8[:, ::1]),
                             ("crds_cc", nb.f8[:, ::1]),
                             ("xl", nb.f8[:, ::1]),
                             ("xh", nb.f8[:, ::1]),
                             ("L", nb.f8[:, ::1]),
                             ("xlnc", nb.f8[::1]),
                             ("xhnc", nb.f8[::1]),
                             ("xlcc", nb.f8[::1]),
                             ("xhcc", nb.f8[::1]),
                             ("dx", nb.f8[::1]),
                             ("min_dx", nb.f8),
                             ("temp_float3_0", nb.f8[::1]),
                             ("temp_float3_1", nb.f8[::1]),
                             ("temp_float3_2", nb.f8[::1]),
                            ])(NbFld)
NbFld_F4_Crds_F4 = jitclass(nb_fld_common +
                            [("data", nb.f4[:, :, :, ::1]),
                             ("crds", nb.f4[:, :, ::1]),
                             ("crds_nc", nb.f4[:, ::1]),
                             ("crds_cc", nb.f4[:, ::1]),
                             ("xl", nb.f4[:, ::1]),
                             ("xh", nb.f4[:, ::1]),
                             ("L", nb.f4[:, ::1]),
                             ("xlnc", nb.f4[::1]),
                             ("xhnc", nb.f4[::1]),
                             ("xlcc", nb.f4[::1]),
                             ("xhcc", nb.f4[::1]),
                             ("dx", nb.f4[::1]),
                             ("min_dx", nb.f4),
                             ("temp_float3_0", nb.f4[::1]),
                             ("temp_float3_1", nb.f4[::1]),
                             ("temp_float3_2", nb.f4[::1]),
                            ])(NbFld)
NbFld_I4_Crds_F8 = jitclass(nb_fld_common +
                            [("data", nb.i4[:, :, :, ::1]),
                             ("crds", nb.f8[:, :, ::1]),
                             ("crds_nc", nb.f8[:, ::1]),
                             ("crds_cc", nb.f8[:, ::1]),
                             ("xl", nb.f8[:, ::1]),
                             ("xh", nb.f8[:, ::1]),
                             ("L", nb.f8[:, ::1]),
                             ("xlnc", nb.f8[::1]),
                             ("xhnc", nb.f8[::1]),
                             ("xlcc", nb.f8[::1]),
                             ("xhcc", nb.f8[::1]),
                             ("dx", nb.f8[::1]),
                             ("min_dx", nb.f8),
                             ("temp_float3_0", nb.f8[::1]),
                             ("temp_float3_1", nb.f8[::1]),
                             ("temp_float3_2", nb.f8[::1]),
                            ])(NbFld)
NbFld_I8_Crds_F8 = jitclass(nb_fld_common +
                            [("data", nb.i8[:, :, :, ::1]),
                             ("crds", nb.f8[:, :, ::1]),
                             ("crds_nc", nb.f8[:, ::1]),
                             ("crds_cc", nb.f8[:, ::1]),
                             ("xl", nb.f8[:, ::1]),
                             ("xh", nb.f8[:, ::1]),
                             ("L", nb.f8[:, ::1]),
                             ("xlnc", nb.f8[::1]),
                             ("xhnc", nb.f8[::1]),
                             ("xlcc", nb.f8[::1]),
                             ("xhcc", nb.f8[::1]),
                             ("dx", nb.f8[::1]),
                             ("min_dx", nb.f8),
                             ("temp_float3_0", nb.f8[::1]),
                             ("temp_float3_1", nb.f8[::1]),
                             ("temp_float3_2", nb.f8[::1]),
                            ])(NbFld)

NbFldTypes = [NbFld_F4_Crds_F4, NbFld_F8_Crds_F8, NbFld_I4_Crds_F8,
              NbFld_I8_Crds_F8]


def make_nb_field(vfield):
    """Construct a Numba Field struct from an existing Viscid Field"""
    vfield = vfield.as_interlaced(force_c_contiguous=True).atleast_3d()
    fld_dtype = np.dtype(vfield.dtype)

    # construct the type-specific NbFld
    if fld_dtype == np.dtype('f4'):
        nb_fld, dat_dtype, crd_dtype = NbFld_F4_Crds_F4(), 'f4', 'f4'
    elif fld_dtype == np.dtype('f8'):
        nb_fld, dat_dtype, crd_dtype = NbFld_F8_Crds_F8(), 'f8', 'f8'
    elif fld_dtype == np.dtype('i4'):
        nb_fld, dat_dtype, crd_dtype = NbFld_I4_Crds_F8(), 'i4', 'f8'
    elif fld_dtype == np.dtype('i8'):
        nb_fld, dat_dtype, crd_dtype = NbFld_I8_Crds_F8(), 'i8', 'f8'
    else:
        raise RuntimeError("Bad field dtype for cython code {0}"
                           "".format(fld_dtype))

    # set the data

    dat = vfield.data
    while len(dat.shape) < 4:
        dat = np.expand_dims(dat, axis=4)
    nb_fld.data = dat.astype(dat_dtype)

    # Now set crd, n, xl, xh, etc.

    sshape = np.array(vfield.sshape)
    sshape_nc = np.array(vfield.crds.shape_nc)
    sshape_cc = np.array(vfield.crds.shape_cc)

    sshape_max = max(sshape)
    sshape_nc_max = max(sshape_nc)
    sshape_cc_max = max(sshape_cc)

    nb_fld.crds = np.nan * np.empty((3, 3, sshape_max), dtype=crd_dtype)
    nb_fld.crds_nc = np.nan * np.empty((3, sshape_nc_max), dtype=crd_dtype)
    nb_fld.crds_cc = np.nan * np.empty((3, sshape_cc_max), dtype=crd_dtype)

    nb_fld.nr_nodes = np.zeros((3,), dtype='int')
    nb_fld.nr_cells = np.zeros((3,), dtype='int')
    nb_fld.n = np.zeros((3,), dtype='int')
    nb_fld.nm1 = np.zeros((3,), dtype='int')
    nb_fld.nm2 = np.zeros((3,), dtype='int')

    nb_fld.xl = np.nan * np.empty((3, 3), dtype=crd_dtype)
    nb_fld.xh = np.nan * np.empty((3, 3), dtype=crd_dtype)
    nb_fld.L = np.nan * np.empty((3, 3), dtype=crd_dtype)
    nb_fld.dx = np.nan * np.empty((3,), dtype=crd_dtype)

    nb_fld.xlnc = np.nan * np.empty((3,), dtype=crd_dtype)
    nb_fld.xhnc = np.nan * np.empty((3,), dtype=crd_dtype)
    nb_fld.xlcc = np.nan * np.empty((3,), dtype=crd_dtype)
    nb_fld.xhcc = np.nan * np.empty((3,), dtype=crd_dtype)

    nb_fld.temp_int3_0 = 99999999 * np.ones((3,), dtype="int")
    nb_fld.temp_int3_1 = 99999999 * np.ones((3,), dtype="int")
    nb_fld.temp_int3_2 = 99999999 * np.ones((3,), dtype="int")
    nb_fld.temp_float3_0 = np.nan * np.empty((3,), dtype=crd_dtype)
    nb_fld.temp_float3_1 = np.nan * np.empty((3,), dtype=crd_dtype)
    nb_fld.temp_float3_2 = np.nan * np.empty((3,), dtype=crd_dtype)

    x, y, z = vfield.get_crds_vector()
    _crd_lst = [[_x, _y, _z] for _x, _y, _z in zip(x, y, z)]

    # xnc, ync, znc = vfield.get_crds_nc()
    # xcc, ycc, zcc = vfield.get_crds_cc()
    # nb_fld.xnc = xnc.astype(crd_dtype, copy=False)
    # nb_fld.ync = ync.astype(crd_dtype, copy=False)
    # nb_fld.znc = znc.astype(crd_dtype, copy=False)
    # nb_fld.xcc = xcc.astype(crd_dtype, copy=False)
    # nb_fld.ycc = ycc.astype(crd_dtype, copy=False)
    # nb_fld.zcc = zcc.astype(crd_dtype, copy=False)

    for ic in range(3):
        for i in range(3):
            for j in range(sshape[i]):
                nb_fld.crds[ic, i, j] = _crd_lst[ic][i][j]
            # fld.xl[ic, i] = fld.crds[ic, i, 0]
            # fld.xh[ic, i] = fld.crds[ic, i, sshape[i] - 1]

    nb_fld.min_dx = np.min(vfield.crds.min_dx_nc)


    for i in range(3):
        nb_fld.xlnc[i] = vfield.crds.xl_nc[i]
        nb_fld.xhnc[i] = vfield.crds.xh_nc[i]
        nb_fld.xlcc[i] = vfield.crds.xl_cc[i]
        nb_fld.xhcc[i] = vfield.crds.xh_cc[i]

        # import pdb; pdb.set_trace()
        nb_fld.nr_nodes[i] = sshape_nc[i]  # len(fld.crds_nc[i])
        nb_fld.nr_cells[i] = sshape_cc[i]  # len(fld.crds_cc[i])
        nb_fld.n[i] = sshape[i]  # len(fld.crds[i])
        nb_fld.nm1[i] = nb_fld.n[i] - 1
        nb_fld.nm2[i] = nb_fld.n[i] - 2

        for ic in range(3):
            nb_fld.xl[ic, i] = nb_fld.crds[ic, i, 0]
            nb_fld.xh[ic, i] = nb_fld.crds[ic, i, max(sshape[i] - 1, 0)]
            nb_fld.L[ic, i] = nb_fld.xh[ic, i] - nb_fld.xl[ic, i]

        # fld.cached_ind[i] = 0  # dangerous for threads :(
        nb_fld.uniform_crds = vfield.crds._TYPE.startswith("uniform")  # pylint: disable=protected-access
        if nb_fld.uniform_crds:
            nb_fld.dx[i] = (nb_fld.xhnc[i] - nb_fld.xlnc[i]) / nb_fld.nr_nodes[i]
        else:
            nb_fld.dx[i] = np.nan

        for ic in range(3):
            if nb_fld.xh[ic, i] < nb_fld.xl[ic, i]:
                raise RuntimeError("Forward crds only in cython code")

    return nb_fld


def _main():
    import viscid
    for dt in ['f8', 'f4', 'i8', 'i4']:
        f0 = viscid.ones((3, 4, 5), dtype=dt)
        f0.data = np.arange(np.prod(f0.shape)).astype(f0.dtype)
        f1 = make_nb_field(f0)
        # f1 = viscid.timeit(make_nb_field, f0)
        print(dt, "->", type(f1), f1.data.dtype, f1.crds.dtype)
    # viscid.interact()
    return 0

if __name__ == "__main__":
    sys.exit(_main())

##
## EOF
##
