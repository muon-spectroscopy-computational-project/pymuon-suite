"""
Author: Simone Sturniolo and Adam Laverack
"""

# Python 2-to-3 compatibility code
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
from datetime import datetime


def write_tensors(tensors, filename, symbols):
    """
    Write out a set of 2 dimensional tensors for every atom in a system.

    | Args:
    |   tensors(Numpy float array, shape: (Atoms, :, :): A list of tensors
    |       for each atom.
    |   filename(str): Filename for file.
    |   symbols(str array): List containing chemical symbol of each atom in
    |       system.
    |
    | Returns: Nothing
    """
    tensfile = open(filename, 'w')
    for i in range(np.size(tensors, 0)):
        tensfile.write('{0} {1}\n'.format(symbols[i], i))
        tensfile.write('\n'.join(['\t'.join([str(x) for x in l])
                                  for l in tensors[i]]) + '\n')


def write_cluster_report(args, params, clusters):

    with open(params['name'] + '_clusters.txt', 'w') as f:

        f.write("""
****************************
|                          |
|         MUAIRSS          |
|    Clustering report     |
|                          |
****************************

Name: {name}
Date: {date}
Structure file(s): {structs}
Parameter file: {param}

*******************

""".format(name=params['name'], date=datetime.now(), structs=args.structures,
           param=args.parameter_file))

        for name, cdata in clusters.items():

            f.write('Clusters for {0}:\n'.format(name))
