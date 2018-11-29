"""
Author: Simone Sturniolo and Adam Laverack
"""

# Python 2-to-3 compatibility code
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np

def write_tensors(tensors, sname, symbols):
    """
    Write out a set of 3x3 tensors for every atom in a system.

    | Args:
    |   tensors(Numpy float array, shape: (Atoms, 3, 3): A list of 3x3 tensors
    |       for each atom.
    |   sname(str): Seedname for file (i.e. filename will be sname_tensors.dat).
    |   symbols(str array): List containing chemical symbol of each atom in
    |       system.
    |
    | Returns: Nothing
    """
    tensfile = open(sname + '_tensors.dat', 'w')
    for i in range(np.size(tensors, 0)):
        tensfile.write('{0} {1}\n'.format(symbols[i], i))
        tensfile.write('\n'.join(['\t'.join([str(x) for x in l]) for l in tensors[i]]) + '\n')
