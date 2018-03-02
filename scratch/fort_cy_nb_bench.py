#!/usr/bin/env python

from __future__ import division, print_function
import cProfile
import os
import subprocess
import sys
import tempfile

import numpy as np
import viscid

from viscid.nb_tools.nbcalc import nb_interp_nearest, nb_interp_trilin
from viscid.nb_tools.nbstreamline import nb_calc_streamlines


# os.environ['NUMBA_NUM_THREADS'] = "2"
# os.environ['NUMBA_WARNINGS'] = "1"
# os.environ['NUMBA_DISABLE_JIT'] = "1"
# os.environ['NUMBA_DUMP_ANNOTATION'] = "1"
# os.environ['NUMBA_DUMP_IR'] = "1"
# os.environ['NUMBA_DUMP_BYTECODE'] = "1"
# os.environ['NUMBA_DUMP_OPTIMIZED'] = "1"
# os.environ['NUMBA_DUMP_ASSEMBLY'] = "1"


# make sure fortran code is built
print_make_output = False
tempfname = tempfile.mkstemp()[1]
with open(tempfname, 'r+') as f:
    ret = subprocess.check_call("make -C fort_tools all", shell=True,
                                stdout=f, stderr=f)
    if ret != 0:
        print("Make failed...")
    if ret != 0 or print_make_output:
        f.seek(0)
        print(f.read())
    if ret != 0:
        sys.exit(ret)
os.remove(tempfname)

from fort_tools import fort_interp_trilin, fort_topology  # pylint: disable=wrong-import-position


N = (128, 64, 64)
XL = (-30, -30, -30)
XH = (+30, +30, +30)
DTYPE = 'f4'


def print_seedup(a_name, a_time, b_name, b_time, prefix=""):
    frac = b_time / a_time
    if frac >= 1:
        assesment = "faster"
    else:
        frac = 1 / frac
        assesment = "slower"
    print("{0}{1} is {2:.2f}x {3} than {4}".format(prefix, a_name, frac, assesment,
                                                   b_name))

def make_grid():
    x, y, z = [np.linspace(_l, _h, _n) for _l, _h, _n in zip(XL, XH, N)]
    x[-1] += 0.1
    return x, y, z

def make_scalar_fld():
    x, y, z = make_grid()
    f0 = viscid.empty((x, y, z), dtype=DTYPE, center='node')
    f0.data = np.arange(np.prod(f0.shape)).astype(f0.dtype)
    seeds = viscid.Volume(xl=XL, xh=XH, n=N)
    return f0, seeds

def make_vector_fld():
    x, y, z = make_grid()
    f0 = viscid.empty((x, y, z), dtype=DTYPE, nr_comps=3, center='node')
    viscid.fill_dipole(f0)
    # seeds = viscid.Sphere(r=10.0, nphi=64, ntheta=32)
    seeds = viscid.Sphere(r=10.0, nphi=32, ntheta=48)
    return f0, seeds


def benchmark_interp_nearest(precompile=True, profile=True):
    which = "interp_nearest"
    print(which)
    print('-' * len(which))

    f0, seeds = make_scalar_fld()

    if precompile:
        print("Compiling Numba", which)
        stats = dict()
        viscid.timeit(nb_interp_nearest, f0, seeds, timeit_repeat=2,
                      timeit_stats=stats, timeit_quiet=True)
        print("Compilation took {0:.3g} sec".format(stats['max'] - stats['min']))

    if profile:
        print("Profiling", which)
        cProfile.runctx('nb_interp_nearest(f0, seeds)', globals(), locals(),
                        filename='interp_nearest.prof')

    print("Timing", which)
    cy_stats, nb_stats = dict(), dict()
    retCY = viscid.timeit(viscid.interp_nearest, f0, seeds,
                          timeit_repeat=10, timeit_stats=cy_stats)
    retNB = viscid.timeit(nb_interp_nearest, f0, seeds,
                          timeit_repeat=10, timeit_stats=nb_stats)
    print_seedup("Numba", nb_stats['min'], "Cython", cy_stats['min'], prefix="@ ")

    assert np.all(f0.data == retCY.data)
    assert np.all(f0.data == retNB.data)

def benchmark_interp_trilin(precompile=True, profile=True):
    which = "interp_trilin"
    print(which)
    print('-' * len(which))

    f0, seeds = make_scalar_fld()

    if precompile:
        print("Compiling Numba", which)
        stats = dict()
        viscid.timeit(nb_interp_trilin, f0, seeds, timeit_repeat=2,
                      timeit_stats=stats, timeit_quiet=True)
        print("Compilation took {0:.3g} sec".format(stats['max'] - stats['min']))

    if profile:
        print("Profiling", which)
        cProfile.runctx('nb_interp_trilin(f0, seeds)', globals(), locals(),
                        filename='interp_trilin.prof')

    print("Timing", which)
    ft_stats, cy_stats, nb_stats = dict(), dict(), dict()
    retFT = viscid.timeit(fort_interp_trilin, f0, seeds,
                          timeit_repeat=10, timeit_stats=ft_stats)
    retCY = viscid.timeit(viscid.interp_trilin, f0, seeds,
                          timeit_repeat=10, timeit_stats=cy_stats)
    retNB = viscid.timeit(nb_interp_trilin, f0, seeds,
                          timeit_repeat=10, timeit_stats=nb_stats)
    print_seedup("Cython", cy_stats['min'], "Fortran", ft_stats['min'], prefix="@ ")
    print_seedup("Numba", nb_stats['min'], "Fortran", ft_stats['min'], prefix="@ ")

    assert np.all(retCY.data == retNB.data)
    assert np.allclose(retCY.data, retFT.data)

def benchmark_streamline(precompile=True, profile=True, scale=True, plot=True):
    which = "streamline"
    print(which)
    print('-' * len(which))

    f0, seeds = make_vector_fld()

    if precompile:
        print("Compiling Numba", which)
        _seeds = viscid.Sphere(r=10.0, nphi=2, ntheta=2)
        stats = dict()
        viscid.timeit(nb_calc_streamlines, f0, _seeds, timeit_repeat=2,
                      timeit_stats=stats, timeit_quiet=True, ibound=3.7, maxit=3,
                      topo_style='generic')
        # print(">>LINES>>")
        # print(_l)
        # print("<<LINES<<")
        print("Compilation took {0:.3g} sec".format(stats['max'] - stats['min']))

    if profile:
        print("Profiling", which)

    print("Timing", which)

    sl_kwargs = dict(ibound=3.7, ds0=0.020)
    lines, _ = viscid.streamlines(f0, seeds, **sl_kwargs)
    nsegs_cython = np.sum([line.shape[1] for line in lines])
    lines = None
    lines, _ = nb_calc_streamlines(f0, seeds, **sl_kwargs)
    nsegs_numba = np.sum([line.shape[1] for line in lines])
    lines = None

    ft_stats, cy_stats, nb_stats = dict(), dict(), dict()
    bench_output_type = viscid.OUTPUT_TOPOLOGY
    sl_kwargs.update(output=bench_output_type)

    retFT = viscid.timeit(fort_topology, f0, seeds, timeit_repeat=6,
                          timeit_stats=ft_stats)
    _, retCY = viscid.timeit(viscid.streamlines, f0, seeds, timeit_repeat=6,
                             timeit_stats=cy_stats, **sl_kwargs)
    _, retNB = viscid.timeit(nb_calc_streamlines, f0, seeds, timeit_repeat=6,
                             timeit_stats=nb_stats, **sl_kwargs)

    fort_per_seg = ft_stats['min'] / retFT.get_info('nsegs')
    cy_per_seg = cy_stats['min'] / nsegs_cython
    nb_per_seg = nb_stats['min'] / nsegs_numba

    print("Segs Fortran", retFT.get_info('nsegs'))
    print("Segs Cython ", nsegs_cython)
    print("Segs Numba ", nsegs_numba)

    print("Fortran took {0:.3g} sec/seg".format(fort_per_seg))
    print("Cython took {0:.3g} sec/seg".format(cy_per_seg))
    print("Numba took {0:.3g} sec/seg".format(nb_per_seg))
    print_seedup("Cython", cy_per_seg, "Fortran", fort_per_seg, prefix="@ ")
    print_seedup("Numba", nb_per_seg, "Fortran", fort_per_seg, prefix="@ ")

    if plot:
        from viscid.plot import mpl
        mpl.clf()
        mpl.subplot(221, projection='polar')
        mpl.plot(retNB, hemisphere='north')
        mpl.subplot(222, projection='polar')
        mpl.plot(retNB, hemisphere='south')
        mpl.subplot(223, projection='polar')
        mpl.plot(retCY, hemisphere='north')
        mpl.subplot(224, projection='polar')
        mpl.plot(retCY, hemisphere='south')
        mpl.show()

    assert np.all(retCY.data == retNB.data)

    if scale:
        thetas = np.logspace(np.log10(3), np.log10(144), 8).astype('i')

        cy_nsegs = [None] * len(thetas)
        fort_nsegs = [None] * len(thetas)
        cy_mintime = [None] * len(thetas)
        fort_mintime = [None] * len(thetas)
        nb_nsegs = [None] * len(thetas)
        nb_mintime = [None] * len(thetas)

        for i, ntheta in enumerate(thetas):
            seeds = viscid.Sphere(r=10.0, ntheta=ntheta, nphi=32)
            _stats = dict()

            topo = viscid.timeit(fort_topology, f0, seeds, timeit_repeat=5,
                                 timeit_stats=_stats, timeit_quiet=True)
            fort_nsegs[i] = topo.get_info('nsegs')
            fort_mintime[i] = _stats['min']

            _, topo = viscid.timeit(viscid.calc_streamlines, f0, seeds,
                                    ibound=3.7, ds0=0.020, output=bench_output_type,
                                    timeit_repeat=5, timeit_stats=_stats,
                                    timeit_quiet=True)
            lines, _ = viscid.streamlines(f0, seeds, ibound=3.7, ds0=0.020)
            cy_nsegs[i] = np.sum([line.shape[1] for line in lines])
            cy_mintime[i] = _stats['min']

            _, topo = viscid.timeit(nb_calc_streamlines, f0, seeds,
                                    ibound=3.7, ds0=0.020, output=bench_output_type,
                                    timeit_repeat=5, timeit_stats=_stats,
                                    timeit_quiet=True)
            lines, _ = nb_calc_streamlines(f0, seeds, ibound=3.7, ds0=0.020)
            nb_nsegs[i] = np.sum([line.shape[1] for line in lines])
            nb_mintime[i] = _stats['min']

        from viscid.plot import mpl
        mpl.clf()
        mpl.plt.plot(nb_nsegs, nb_mintime, label="Numba")
        mpl.plt.plot(cy_nsegs, cy_mintime, label="Cython")
        mpl.plt.plot(fort_nsegs, fort_mintime, label="Fortran")
        mpl.plt.legend(loc=0)
        mpl.plt.xlabel('Number of segments calculated')
        mpl.plt.ylabel('time to calculate')
        mpl.show()

        mpl.clf()
        nb_tperseg = np.array(nb_mintime) / np.array(nb_nsegs)
        cy_tperseg = np.array(cy_mintime) / np.array(cy_nsegs)
        fort_tperseg = np.array(fort_mintime) / np.array(fort_nsegs)
        mpl.plt.plot(thetas, nb_tperseg / fort_tperseg, label="over numba")
        mpl.plt.plot(thetas, cy_tperseg / fort_tperseg, label="over cython")
        mpl.plt.xlabel('ntheta')
        mpl.plt.ylabel('Fortran Speedup')
        mpl.show()

def main():
    # np.seterr(all='raise')
    precompile = True
    profile = False
    scale = False
    plot = False

    benchmark_interp_nearest(precompile=precompile, profile=profile)
    print()
    benchmark_interp_trilin(precompile=precompile, profile=profile)
    print()
    benchmark_streamline(precompile=precompile, profile=profile, scale=scale,
                         plot=plot)
    print()

    return 0

if __name__ == "__main__":
    sys.exit(main())

##
## EOF
##
