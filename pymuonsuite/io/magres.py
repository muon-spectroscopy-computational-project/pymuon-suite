# Python 2-to-3 compatibility code
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np

from ase.io.magres import read_magres

def parse_hyperfine_magres(infile):

    # First, `simply parse the magres file via ASE
    mgr = read_magres(infile, True)

    # Now go for the magres_old block

    if 'magresblock_magres_old' not in mgr.info:
        raise RuntimeError('.magres file has no hyperfine information')

    hfine = parse_hyperfine_oldblock(mgr.info['magresblock_magres_old'])

    labels, indices = mgr.get_array('labels'), mgr.get_array('indices')

    hfine_array = []
    for l, i in zip(labels, indices):
        hfine_array.append(hfine[l][i])

    mgr.new_array('hyperfine', np.array(hfine_array))

    return mgr


def parse_hyperfine_oldblock(block):

    hfine_dict = {}

    sp = None
    n = None
    tens = None
    block_lines = block.split('\n')
    for i, l in enumerate(block_lines):
        if 'Atom:' in l:
            # Get the species and index
            _, sp, n = l.split()
            n = int(n)
        if 'TOTAL tensor' in l:
            tens = np.array([[float(x) for x in row.split()]
                             for row in block_lines[i+2:i+5]])
            # And append
            if sp is None:
                raise RuntimeError('Invalid block in magres hyperfine file')
            if sp not in hfine_dict:
                hfine_dict[sp] = {}
            hfine_dict[sp][n] = tens

    return hfine_dict
