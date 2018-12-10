"""
Author: Adam Laverack and Simone Sturniolo
"""

# Python 2-to-3 compatibility code
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import shutil

import numpy as np
import scipy.constants as cnst
from ase import Atoms
from ase import io as ase_io
from soprano.collection import AtomsCollection
from soprano.utils import seedname

from pymuonsuite.io.castep import parse_castep_muon, parse_final_energy
from pymuonsuite.io.magres import parse_hyperfine_magres
from pymuonsuite.io.output import write_tensors
from pymuonsuite.quantum.vibrational.grid import calc_wavefunction, weighted_tens_avg
from pymuonsuite.quantum.vibrational.output import hfine_report
from pymuonsuite.quantum.vibrational.phonons import ase_phonon_calc, calc_harm_potential
from pymuonsuite.quantum.vibrational.phonons import get_major_emodes
from pymuonsuite.utils import create_displaced_cells, find_ipso_hydrogen
try:
    from casteppy.data.phonon import PhononData
except ImportError:
    raise ImportError("""
Can't use castep phonon interface due to casteppy not being installed.
Please download and install casteppy from Bitbucket:

HTTPS:  https://bitbucket.org/casteppy/casteppy.git
SSH:    git@bitbucket.org:casteppy/casteppy.git

and try again.""")

def muon_harmonic(cell_f, mu_sym, grid_n, property, value_type, weight_type,
                pname=None, ignore_ipsoH=False, solver=False, args_w=False,
                ase_phonons=False, dftb_phonons=True):
    """
    Given a file containing phonon modes of a muonated molecule, write
    out a set of structure files with the muon progressively displaced in
    grid_n increments along each axis of the phonon modes, creating a grid.
    Alternatively, read in coupling values calculated at each point of a grid
    created using this function's write mode and average them to give an estimate
    of the actual coupling values accounting for nuclear quantum effects.

    | Args:
    |   cell_f (str): Path to structure file (e.g. .cell file for CASTEP)
    |   mu_sym (str): Symbol used to represent muon in structure file
    |   grid_n (int): Number of increments to make along each phonon axis
    |   property(str): Property to be calculated. Currently accepted values:
    |       "hyperfine" (hyperfine tensors),
    |   pname (str): Path of param file which will be copied into folders
    |       along with displaced cell files for convenience
    |   ignore_ipsoH (bool): If true, ignore ipso hydrogen calculations
    |   solver (bool): If true, use qlab (only if installed) to numerically
    |       calculate the harmonic wavefunction
    |   args_w (bool): Write files if true, parse if false
    |   ase_phonons(bool): If true, use ASE to calculate phonon modes. ASE will
    |       use the calculator of the input cell, e.g. CASTEP for .cell files. Set
    |       dftb_phonons to True in order to use dftb+ as the calculator instead.
    |       If false, will read in CASTEP phonons.
    |   dftb_phonons(bool): Use dftb+ with ASE to calculate phonons if true.
    |       Requires ase_phonons set to true.
    |
    | Returns: Nothing
    """
    cell = ase_io.read(cell_f)
    sname = seedname(cell_f)
    #Parse muon data
    mu_index, iH_index, mu_mass = parse_castep_muon(cell, mu_sym, ignore_ipsoH)

    if ase_phonons:
        #Calculate phonons using ASE
        masses = cell.get_masses()
        masses[mu_index] = mu_mass/cnst.u
        cell.set_masses(masses)
        evals, evecs = ase_phonon_calc(cell, dftb_phonons)
    else:
        # Parse CASTEP phonon data into casteppy object
        pd = PhononData(sname)
        # Convert frequencies back to cm-1
        pd.convert_e_units('1/cm')
        # Get phonon frequencies+modes
        evals = pd.freqs
        evecs = pd.eigenvecs

    # Get muon phonon modes
    mu_evecs_index, mu_evecs, mu_evecs_ortho = get_major_emodes(evecs[0], mu_index)
    # Get muon phonon frequencies and convert to radians/second
    mu_evals = np.array(evals[0][mu_evecs_index]*1e2*cnst.c*np.pi*2)
    # Displacement in Angstrom
    R = np.sqrt(cnst.hbar/(mu_evals*mu_mass))*1e10


    # Write cells with displaced muon
    if args_w:
        for i, Ri in enumerate(R):
            cell.info['name'] = sname + '_' + str(i+1)
            dirname = '{0}_{1}'.format(sname, i+1)
            lg = create_displaced_cells(cell, mu_index, grid_n, 3*mu_evecs[i]*Ri)
            collection = AtomsCollection(lg)
            for atom in collection:
                atom.set_calculator(cell.calc)
            collection.save_tree(dirname, "cell")
            #Copy parameter file if specified
            if pname:
                for j in range(grid_n):
                    shutil.copy(pname, os.path.join(dirname,
                         '{0}_{1}_{2}/{0}_{1}_{2}.param'.format(sname, i+1, j)))

    else:
        E_table = []
        hfine_table = ipso_hfine_table = np.zeros((np.size(R), grid_n))
        num_species = np.size(cell.get_array('castep_custom_species'))
        if value_type == 'scalar':
            grid_tensors = np.zeros((num_species, np.size(R), grid_n, 1, 1))
        elif value_type == 'vector':
            grid_tensors = np.zeros((num_species, np.size(R), grid_n, 1, 3))
        elif value_type == 'matrix':
            grid_tensors = np.zeros((num_species, np.size(R), grid_n, 3, 3))

        # Parse tensors from appropriate files and energy from .castep files
        for i, Ri in enumerate(R):
            E_table.append([])
            dirname = '{0}_{1}'.format(sname, i+1)
            for j in range(grid_n):
                if property == 'hyperfine':
                    tensor_file = os.path.join(dirname,
                        '{0}_{1}_{2}/{0}_{1}_{2}.magres'.format(sname, i+1, j))
                    tensors = (parse_hyperfine_magres(tensor_file)).get_array('hyperfine')
                else:
                    raise ValueError("Invalid value for property")
                for k, tensor in enumerate(tensors):
                    grid_tensors[k][i][j][:][:] = tensor
                castf = os.path.join(dirname,
                    '{0}_{1}_{2}/{0}_{1}_{2}.castep'.format(sname, i+1, j))
                E_table[-1].append(parse_final_energy(castf))

        E_table = np.array(E_table)
        if (E_table.shape != (np.size(R), grid_n)):
            raise RuntimeError("Incomplete or absent castep data")

        symbols = cell.get_array('castep_custom_species')

        #Calculate vibrational average of property and write it out
        if weight_type == 'harmonic':
            weighting = calc_wavefunction(R, grid_n, write_table = True, sname = sname)
        else:
            raise ValueError("Invalid value for weight")
        tens_avg = weighted_tens_avg(grid_tensors, weighting)
        write_tensors(tens_avg, sname, symbols)
        calc_harm_potential(R, grid_n, mu_mass, mu_evals, E_table, sname)

        if property == 'hyperfine':
            hfine_report(R, grid_n, grid_tensors[mu_index], tens_avg[mu_index],
                weighting, sname, mu_sym)
            if not ignore_ipsoH:
                hfine_report(R, grid_n, grid_tensors[iH_index], tens_avg[iH_index],
                weighting, sname, "{0} {1} (ipso)".format(symbols[iH_index], iH_index))

    return
