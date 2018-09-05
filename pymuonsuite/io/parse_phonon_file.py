# Python 2-to-3 compatibility code
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
from ase import Atoms

def parse_phonon_file(phfile):

    lines = open(phfile).readlines()

    def parse_pos(block):
        posblock = {'pos': [], 'sym': [], 'm': []}
        for l in block:
            l_spl = l.split()
            posblock['pos'].append(map(float, l_spl[1:4]))
            posblock['sym'].append(l_spl[4])
            posblock['m'].append(float(l_spl[5]))

        return posblock

    def parse_modes(block, num_ions, num_modes):
        evecs = [[None for i in range(num_ions)] for m in range(num_modes)]
        for l in block:
            l_spl = l.split()
            m, i = map(int, l_spl[:2])
            evecs[m-1][i-1] = [float(l_spl[j])+1.0j*float(l_spl[j+1]) for j in range(2,8,2)]

        return evecs

    num_ions = 0
    num_modes = 0
    cell = None
    posblock = None
    phonon_freq = None
    phonon_modes = None
    for i in range(len(lines)):
        l = lines[i]
        # First, grab cell
        if "Number of ions" in l:
            num_ions = int(l.split()[-1])
        if "Number of branches" in l:
            num_modes = int(l.split()[-1])
        if "Unit cell vectors (A)" in l:
            cell = [[float(x) for x in e.split()] for e in lines[i+1:i+4]]
        if  "Fractional Co-ordinates" in l:
            posblock = parse_pos(lines[i+1:i+num_ions+1])
        if "q-pt=" in l:
            # Phonon frequency parsing time!
            phonon_freq = [float(m.split()[-1]) for m in lines[i+1:i+num_modes+1]]
        if "Phonon Eigenvectors" in l:
            # Mode eigenvectors time
            phonon_modes = parse_modes(lines[i+2:i+num_modes*num_ions+2], num_ions, num_modes)

    # Did it all go well?
    if None in (num_ions, num_modes, cell, posblock, phonon_freq, phonon_modes):
        raise RuntimeError('Invalid phonon file {0}'.format(phfile))

    # Now make it into an ASE atoms object
    struct = Atoms([s.split(':')[0] for s in posblock['sym']],
                   cell=cell,
                   scaled_positions=posblock['pos'],
                   masses=posblock['m'])
    struct.set_array('castep_custom_species', np.array(posblock['sym']))

    return struct, np.array(phonon_freq), np.array(phonon_modes)