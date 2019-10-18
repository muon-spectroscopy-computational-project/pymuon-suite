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

            for calc, clusts in cdata.items():

                # Computer readable
                fdat = open(params['name'] +
                            '_{0}_{1}_clusters.dat'.format(name, calc), 'w')

                f.write('CALCULATOR: {0}\n'.format(calc))
                (cinds, cgroups), ccolls, gvecs = clusts

                f.write('\t{0} clusters found\n'.format(max(cinds)))

                for i, g in enumerate(cgroups):

                    f.write(
                        '\n\n\t-----------\n\tCluster '
                        '{0}\n\t-----------\n'.format(i+1))
                    f.write('\tStructures: {0}\n'.format(len(g)))
                    coll = ccolls[i+1]
                    E = gvecs[g, 0]
                    Emin = np.amin(E)
                    Eavg = np.average(E)
                    Estd = np.std(E)

                    f.write('\n\tEnergy (eV):\n')
                    f.write('\tMinimum\t\tAverage\t\tStDev\n')
                    f.write('\t{0:.2f}\t\t{1:.2f}\t\t{2:.2f}\n'.format(Emin,
                                                                       Eavg,
                                                                       Estd))

                    fdat.write('\t'.join(map(str, [i+1, len(g),
                                                   Emin, Eavg, Estd])) + '\n')

                    f.write('\n\n\tStructure list:')

                    for j, s in enumerate(coll):
                        if j % 4 == 0:
                            f.write('\n\t')
                        f.write('{0}\t'.format(s.info['name']))

                fdat.close()

                # Print distance matrix

                f.write('\n\n\t----------\n\n\tSimilarity (ranked):\n')

                centers = np.array([np.average(gvecs[g], axis=0)
                                    for g in cgroups])
                dmat = np.linalg.norm(
                    centers[:, None]-centers[None, :], axis=-1)

                inds = np.triu_indices(len(cgroups), k=1)
                for i in np.argsort(dmat[inds]):
                    c1 = inds[0][i]
                    c2 = inds[1][i]
                    d = dmat[c1, c2]
                    f.write('\t{0} <--> {1} (distance = {2:.3f})\n'.format(c1,
                                                                           c2,
                                                                           d))

            f.write('\n--------------------------\n\n')

        f.write('\n==========================\n\n')
