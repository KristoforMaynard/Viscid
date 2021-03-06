#!/usr/bin/env python

from __future__ import division, print_function
import os
import sys

import viscid
from viscid.plot import vpyplot as vlt

def main():
    f = viscid.load_file("~/dev/work/tmedium/*.3d.[-1].xdmf")
    grid = f.get_grid()

    gslc = "x=-26j:12.5j, y=-15j:15j, z=-15j:15j"
    # gslc = "x=-12.5j:26j, y=-15j:15j, z=-15j:15j"

    b_cc = f['b_cc'][gslc]
    b_cc.name = "b_cc"
    b_fc = f['b_fc'][gslc]
    b_fc.name = "b_fc"

    e_cc = f['e_cc'][gslc]
    e_cc.name = "e_cc"
    e_ec = f['e_ec'][gslc]
    e_ec.name = "e_ec"

    pp = f['pp'][gslc]
    pp.name = 'pp'

    pargs = dict(logscale=True, earth=True)

    # vlt.clf()
    # ax1 = vlt.subplot(211)
    # vlt.plot(f['pp']['y=0j'], **pargs)
    # # vlt.plot(viscid.magnitude(f['b_cc']['y=0j']), **pargs)
    # # vlt.show()
    # vlt.subplot(212, sharex=ax1, sharey=ax1)
    # vlt.plot(viscid.magnitude(viscid.fc2cc(f['b_fc'])['y=0j']), **pargs)
    # vlt.show()

    basename = './tmediumR.3d.{0:06d}'.format(int(grid.time))
    viscid.save_fields(basename + '.h5', [b_cc, b_fc, e_cc, e_ec, pp])

    f2 = viscid.load_file(basename + ".xdmf")

    pargs = dict(logscale=True, earth=True)

    vlt.clf()
    ax1 = vlt.subplot(211)
    vlt.plot(f2['pp']['y=0j'], style='contour', levels=5, colorbar=None,
             colors='k', **pargs)
    vlt.plot(viscid.magnitude(f2['b_cc']['y=0j']), **pargs)
    vlt.subplot(212, sharex=ax1, sharey=ax1)
    vlt.plot(viscid.magnitude(viscid.fc2cc(f2['b_fc'])['y=0j']), **pargs)
    vlt.show()

    os.remove(basename + '.h5')
    os.remove(basename + '.xdmf')

    return 0

if __name__ == "__main__":
    sys.exit(main())

##
## EOF
##
