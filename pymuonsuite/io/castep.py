# Python 2-to-3 compatibility code
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import re
import os
import yaml
import numpy as np
from ase import Atoms
from ase import io
from ase.calculators.castep import Castep
from scipy.constants import physical_constants as pcnst

from pymuonsuite.utils import list_to_string


def save_muonconf_castep(a, folder, params):
    # Muon mass and gyromagnetic ratio
    mass_block = 'AMU\n{0}       0.1138'
    gamma_block = 'radsectesla\n{0}        851586494.1'

    if isinstance(a.calc, Castep):
        ccalc = a.calc
    else:
        ccalc = Castep(castep_command=params['castep_command'])

    ccalc.cell.kpoint_mp_grid.value = list_to_string(params['k_points_grid'])
    ccalc.cell.species_mass = mass_block.format(params['mu_symbol']
                                                ).split('\n')
    ccalc.cell.species_gamma = gamma_block.format(params['mu_symbol']
                                                  ).split('\n')
    ccalc.cell.fix_all_cell = True  # To make sure for older CASTEP versions

    a.set_calculator(ccalc)

    name = os.path.split(folder)[-1]
    io.write(os.path.join(folder, '{0}.cell'.format(name)), a)
    ccalc.atoms = a

    if params['castep_param'] is not '':
        castep_params = yaml.load(open(params['castep_param'], 'r'))
    else:
        castep_params = {}

    castep_params['task'] = "GeometryOptimization"

    # Parameters from .yaml will overwrite parameters from .param
    castep_params['geom_max_iter'] = params['geom_steps']
    castep_params['geom_force_tol'] = params['geom_force_tol']
    castep_params['max_scf_cycles'] = params['max_scc_steps']

    parameter_file = os.path.join(folder, '{0}.param'.format(name))
    yaml.safe_dump(castep_params, open(parameter_file, 'w'),
                   default_flow_style=False)


def parse_castep_ppots(cfile):

    clines = open(cfile).readlines()

    # Find pseudopotential blocks
    ppot_heads = filter(lambda x: 'Pseudopotential Report' in x[1],
                        enumerate(clines))
    ppot_blocks_raw = []

    for pph in ppot_heads:
        i, _ = pph
        for j, l in enumerate(clines[i:]):
            if 'Author:' in l:
                break
        ppot_blocks_raw.append(clines[i:i+j])

    # Now on to actually parse them
    ppot_blocks = {}

    el_re = re.compile(r'Element:\s+([a-zA-Z]{1,2})\s+'
                       r'Ionic charge:\s+([0-9.]+)')
    rc_re = re.compile(r'(?:[0-9]+|loc)\s+[0-9]\s+[\-0-9.]+\s+([0-9.]+)')
    bohr = pcnst['Bohr radius'][0]*1e10

    for ppb in ppot_blocks_raw:
        el = None
        q = None
        rcmin = np.inf
        for l in ppb:
            el_m = el_re.search(l)
            if el_m is not None:
                el, q = el_m.groups()
                q = float(q)
                continue
            rc_m = rc_re.search(l)
            if rc_m is not None:
                rc = float(rc_m.groups()[0])*bohr
                rcmin = min(rc, rcmin)
        ppot_blocks[el] = (q, rcmin)

    return ppot_blocks

def parse_final_energy(infile):
    E = None
    for l in open(infile).readlines():
        if "Final energy" in l:
            try:
                E = float(l.split()[3])
            except ValueError:
                raise RuntimeError(
                    "Corrupt .castep file found: {0}".format(infile))
    return E

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
