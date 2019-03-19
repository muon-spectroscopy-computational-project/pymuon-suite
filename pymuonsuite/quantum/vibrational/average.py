"""
average.py

Quantum vibrational averages
"""

# Python 2-to-3 compatibility code
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import numpy as np
from copy import deepcopy

from ase import io
from ase.io.castep import read_param
from ase.calculators.castep import Castep
from ase.calculators.dftb import Dftb
from soprano.utils import seedname
from soprano.collection import AtomsCollection
try:
    from casteppy.data.phonon import PhononData
except ImportError:
    raise ImportError("""
Can't use castep phonon interface due to casteppy not being installed.
Please download and install casteppy from Bitbucket:

HTTPS:  https://bitbucket.org/casteppy/casteppy.git
SSH:    git@bitbucket.org:casteppy/casteppy.git

and try again.""")

# Internal imports
from pymuonsuite.io.castep import parse_castep_masses, castep_write_input, add_to_castep_block
from pymuonsuite.quantum.vibrational.phonons import ase_phonon_calc
from pymuonsuite.quantum.vibrational.schemes import (IndependentDisplacements,)


class MuonAverageError(Exception):
    pass


def read_castep_phonons(seed, path='.'):
    # Parse CASTEP phonon data into casteppy object
    pd = PhononData(seed, path=path)
    # Convert frequencies back to cm-1
    pd.convert_e_units('1/cm')
    # Get phonon frequencies+modes
    evals = np.array(pd.freqs.magnitude)
    evecs = np.array(pd.eigenvecs)

    # Only grab the gamma point!
    gamma_i = None
    for i, q in enumerate(pd.qpts):
        if np.isclose(q, [0, 0, 0]).all():
            gamma_i = i
            break

    if gamma_i is None:
        raise MuonAverageError('Could not find gamma point phonons in CASTEP'
                               ' phonon file')

    return evals[gamma_i], evecs[gamma_i]


def compute_dftbp_phonons(atoms, param_set, kpts):

    # We put this here so the warning about citations is printed only if
    # necessary
    from pymuonsuite.data.dftb_pars import DFTBArgs

    args = DFTBArgs(param_set).args
    calc = Dftb(label=atoms.info.get('name', 'dftbphonons'),
                atoms=atoms, kpts=kpts, **args)
    atoms.set_calculator(calc)

    evals, evecs, atoms = ase_phonon_calc(atoms, force_clean=True)

    return evals[0], evecs[0], atoms


def create_hfine_castep_calculator(mu_symbol='H:mu', calc=None, param_file=None,
                                   kpts=[1, 1, 1]):
    """Create a calculator containing all the necessary parameters
    for a hyperfine calculation."""

    if not isinstance(calc, Castep):
        calc = Castep()
    else:
        calc = deepcopy(calc)

    gamma_block = calc.cell.species_gamma.value
    calc.cell.species_gamma = add_to_castep_block(gamma_block, mu_symbol,
                                                  851586494.1, 'gamma')

    calc.cell.kpoint_mp_grid = kpts

    if param_file is not None:
        calc.param = read_param(param_file)

    calc.param.task = 'Magres'
    calc.param.magres_task = 'Hyperfine'

    return calc


def muon_vibrational_average_write(cell_file, method='independent', mu_index=-1,
                                   mu_symbol='H:mu', grid_n=20, sigma_n=3,
                                   avgprop='hyperfine', phonon_source='castep',
                                   **kwargs):
    """
    Write input files to compute a vibrational average for a quantity on a muon 
    in a given system.

    | Pars:
    |   cell_file (str):    Filename for input structure file
    |   method (str):       Method to use for the average. Options are 'independent',
    |                       'thermal'. Default is 'independent'.
    |   mu_index (int):     Position of the muon in the given cell file. 
    |                       Default is -1.
    |   mu_symbol (str):    Use this symbol to look for the muon among
    |                       CASTEP custom species. Overrides muon_index if 
    |                       present in cell.
    |   grid_n (int):       Number of configurations used for sampling. Applies slightly
    |                       differently to different schemes.
    |   sigma_n (int):      Number of sigmas of the harmonic wavefunction used
    |                       for sampling.
    |   avgprop (str):      Property to calculate and average. Default is 'hyperfine'.
    |   phonon_source (str):Source of the phonon data. Can be 'castep' or 'asedftbp'.
    |                       Default is 'castep'.
    |   **kwargs:           Other arguments (such as specific arguments for the given 
    |                       phonon method)
    """

    # Open the structure file
    cell = io.read(cell_file)
    path = os.path.split(cell_file)[0]
    sname = seedname(cell_file)
    num_atoms = len(cell)

    cell.info['name'] = sname

    # Fetch species
    try:
        species = cell.get_array('castep_custom_species')
    except KeyError:
        species = np.array(cell.get_chemical_symbols())

    mu_indices = np.where(species == mu_symbol)[0]
    if len(mu_indices) > 1:
        raise MuonAverageError(
            'More than one muon found in the system')
    elif len(mu_indices) == 1:
        mu_index = mu_indices[0]
    else:
        species = list(species)
        species[mu_index] = mu_symbol
        species = np.array(species)

    cell.set_array('castep_custom_species', species)

    # Fetch masses
    masses = parse_castep_masses(cell)

    # Load the phonons
    if phonon_source == 'castep':
        ph_evals, ph_evecs = read_castep_phonons(sname, path)
    elif phonon_source == 'asedftbp':
        ph_evals, ph_evecs, cell = compute_dftbp_phonons(cell,
                                                         kwargs['asedftbp_pars'],
                                                         kwargs['asedftbp_kpts'])
        # Save the optimised file
        fname, ext = os.path.splitext(cell_file)
        io.write(fname + '_opt' + ext, cell)

    # Now create the distribution scheme
    if method == 'independent':
        displsch = IndependentDisplacements(ph_evals, ph_evecs, masses,
                                            mu_index)
        displsch.recalc_displacements(n=grid_n, sigma_n=sigma_n)

    # Make it a collection
    pos = cell.get_positions()
    displaced_cells = []
    for i, d in enumerate(displsch.displacements):
        dcell = cell.copy()
        dcell.set_positions(pos + d)
        dcell.info['name'] = sname + '_displaced_{0}'.format(i)
        displaced_cells.append(dcell)

    # Get a calculator
    if avgprop == 'hyperfine':
        calc = create_hfine_castep_calculator(mu_symbol=mu_symbol,
                                              calc=cell.calc,
                                              param_file=kwargs['castep_out_param'],
                                              kpts=kwargs['castep_out_kpts'])

    displaced_coll = AtomsCollection(displaced_cells)
    displaced_coll.info['displacement_scheme'] = displsch
    displaced_coll.save_tree(sname + '_displaced', castep_write_input,
                             opt_args={'calc': calc})
