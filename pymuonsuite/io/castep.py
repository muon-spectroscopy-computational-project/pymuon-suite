# Python 2-to-3 compatibility code
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import re
import os
import yaml
import numpy as np
import scipy.constants as cnst
from ase import Atoms
from ase import io
from ase.io.castep import write_param
from ase.calculators.castep import Castep
from soprano.selection import AtomSelection


from pymuonsuite.utils import list_to_string
from pymuonsuite.utils import find_ipso_hydrogen


class CastepError(Exception):
    pass


def castep_write_input(a, folder, name=None):
    """Writes input files for an Atoms object with a Castep
    calculator attached.

    | Args:
    |   a (ase.Atoms):  Atoms object to write. Must have a Castep
    |                   calculator attached to carry cell/param
    |                   keywords.
    |   folder (str):   Path to save the input files to.
    |   name (str):     Seedname to save the files with. If not
    |                   given, use the name of the folder.
    """

    if name is None:
        name = os.path.split(folder)[-1]  # Same as folder name

    if not isinstance(a.calc, Castep):
        a = a.copy()
        calc = Castep(atoms=a)
        a.set_calculator(calc)

    io.write(os.path.join(folder, name + '.cell'), a)
    write_param(os.path.join(folder, name + '.param'),
                a.calc.param, force_write=True)


def save_muonconf_castep(a, folder, params):
    # Muon mass and gyromagnetic ratio
    mass_block = 'AMU\n{0}       0.1138'
    gamma_block = 'radsectesla\n{0}        851586494.1'

    if isinstance(a.calc, Castep):
        ccalc = a.calc
    else:
        ccalc = Castep()

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

    if params['castep_param'] is not None:
        castep_params = yaml.load(open(params['castep_param'], 'r'))
    else:
        castep_params = {}

    # Parameters from .yaml will overwrite parameters from .param
    castep_params['task'] = "GeometryOptimization"
    castep_params['geom_max_iter'] = params['geom_steps']
    castep_params['geom_force_tol'] = params['geom_force_tol']
    castep_params['max_scf_cycles'] = params['max_scc_steps']

    parameter_file = os.path.join(folder, '{0}.param'.format(name))
    yaml.safe_dump(castep_params, open(parameter_file, 'w'),
                   default_flow_style=False)


def parse_castep_bands(infile, header=False):
    """Parse eigenvalues from a CASTEP .bands file. This only works with spin
    components = 1.

    | Args:
    |   infile(str): Directory of bands file.
    |   header(bool, default=False): If true, just return the number of k-points
    |       and eigenvalues. Else, parse and return the band structure.
    | Returns:
    |   n_kpts(int), n_evals(int): Number of k-points and eigenvalues.
    |   bands(Numpy float array, shape:(n_kpts, n_evals)): Energy eigenvalues of
    |       band structure.
    """
    file = open(infile, "r")
    lines = file.readlines()
    n_kpts = int(lines[0].split()[-1])
    n_evals = int(lines[3].split()[-1])
    if header == True:
        return n_kpts, n_evals
    if int(lines[1].split()[-1]) != 1:
        raise ValueError("""Either incorrect file format detected or greater
                            than 1 spin component used (parse_castep_bands
                            only works with 1 spin component.)""")
    # Parse eigenvalues
    bands = np.zeros((n_kpts, n_evals))
    for kpt in range(n_kpts):
        for eval in range(n_evals):
            bands[kpt][eval] = float(lines[11+eval+kpt*(n_evals+2)].strip())
    return bands


def parse_castep_mass_block(mass_block):
    """Parse CASTEP custom species masses, returning a dictionary of masses
    by species, in amu.

    | Args:
    |   mass_block (str):   Content of a species_mass block
    | Returns:
    |   masses (dict):      Dictionary of masses by species symbol
    """

    mass_tokens = [l.split() for l in mass_block.split('\n')]
    custom_masses = {}

    units = {
        'amu': 1,
        'm_e': cnst.m_e/cnst.u,
        'kg': 1.0/cnst.u,
        'g': 1e-3/cnst.u
    }

    # Is the first line a unit?
    u = 1
    if len(mass_tokens) > 0 and len(mass_tokens[0]) == 1:
        try:
            u = units[mass_tokens[0][0]]
        except KeyError:
            raise CastepError('Invalid mass unit in species_mass block')

        mass_tokens.pop(0)

    for tk in mass_tokens:
        try:
            custom_masses[tk[0]] = float(tk[1])*u
        except (ValueError, IndexError):
            raise CastepError('Invalid line in species_mass block')

    return custom_masses


def parse_castep_masses(cell):
    """Parse CASTEP custom species masses, returning an array of all atom masses
    in .cell file with corrected custom masses.

    | Args:
    |   cell(ASE Atoms object): Atoms object containing relevant .cell file
    | Returns:
    |   masses(Numpy float array, shape(no. of atoms)): Correct masses of all
    |       atoms in cell file.
    """
    mass_block = cell.calc.cell.species_mass.value
    if mass_block is None:
        return cell.get_masses()

    custom_masses = parse_castep_mass_block(mass_block)

    masses = cell.get_masses()
    elems = cell.get_chemical_symbols()
    elems = cell.arrays.get('castep_custom_species', elems)

    masses = [custom_masses.get(elems[i], m) for i, m in enumerate(masses)]

    cell.set_masses(masses)

    return masses


def parse_castep_gamma_block(gamma_block):
    """Parse CASTEP custom species gyromagnetic ratios, returning a 
    dictionary of gyromagnetic ratios by species, in radsectesla.

    | Args:
    |   gamma_block (str):   Content of a species_gamma block
    | Returns:
    |   gammas (dict):      Dictionary of gyromagnetic ratios by species symbol
    """

    gamma_tokens = [l.split() for l in gamma_block.split('\n')]
    custom_gammas = {}

    units = {
        'agr': cnst.e/cnst.m_e,
        'radsectesla': 1,
        'mhztesla': 0.5e-6/np.pi,
    }

    # Is the first line a unit?
    u = 1
    if len(gamma_tokens) > 0 and len(gamma_tokens[0]) == 1:
        try:
            u = units[gamma_tokens[0][0]]
        except KeyError:
            raise CastepError('Invalid gamma unit in species_gamma block')

        gamma_tokens.pop(0)

    for tk in gamma_tokens:
        try:
            custom_gammas[tk[0]] = float(tk[1])*u
        except (ValueError, IndexError):
            raise CastepError('Invalid line in species_gamma block')

    return custom_gammas


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
    bohr = cnst.physical_constants['Bohr radius'][0]*1e10

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
    """
    Parse final energy from .castep file

    | Args:
    |   infile (str): Directory of .castep file
    |
    | Returns:
    |   E (float): Value of final energy
    """
    E = None
    for l in open(infile).readlines():
        if "Final energy" in l:
            try:
                E = float(l.split()[3])
            except ValueError:
                raise RuntimeError(
                    "Corrupt .castep file found: {0}".format(infile))
    return E
