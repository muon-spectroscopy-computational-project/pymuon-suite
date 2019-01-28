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
import random

import numpy as np
import scipy.constants as cnst
from ase import Atoms
from ase import io as ase_io
from soprano.collection import AtomsCollection
from soprano.selection import AtomSelection
from soprano.utils import seedname

from pymuonsuite.io.castep import parse_castep_muon, parse_castep_masses
from pymuonsuite.io.castep import parse_final_energy
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

def vib_avg(cell_f, method, mu_sym, grid_n, property, value_type, atoms_ind=[],
                weight_type='harmonic', pname=None, solver=False, args_w=False,
                ase_phonons=False, dftb_phonons=True):
    """
    (Write mode) Given a file containing phonon modes of a molecule, write
    out a set of structure files, one for each major mode of each atom,
    with the atoms progressively displaced in grid_n increments along each axis
    of their major phonon modes, creating a grid of displacements for each atom.
    (Read mode) Read in coupling values calculated at each point of a grid
    created using this function's write mode and average them to give an estimate
    of the actual coupling values accounting for nuclear quantum effects for each
    atom displaced.

    | Args:
    |   cell_f (str): Path to structure file (e.g. .cell file for CASTEP)
    |   mu_sym (str): Symbol used to represent muon in structure file
    |   grid_n (int): Number of increments to make along each phonon axis
    |   atoms_ind(int array): Array of indices of atoms to be vibrated, counting
    |       from 1. E.g. for first 3 atoms in cell file enter [1, 2, 3].
    |       Enter [-1] to select all atoms.
    |   property(str): Property to be calculated. Currently accepted values:
    |       "hyperfine" (hyperfine tensors),
    |   value_type(str): Is value being calculated a 'matrix', 'vector', or
    |       'scalar'? (e.g. hyperfine tensor is a matrix)
    |   weight_type(str): Type of weighting to be used, currently accepted
    |       values: "harmonic" (harmonic oscillator wavefunction)
    |   pname (str): Path of param file which will be copied into folders
    |       along with displaced cell files for convenience
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
    #Select all atoms if -1 input
    if atoms_ind[0] == -1:
        atoms_ind = np.arange(1, 50, 1, int)

    #Parse cell data
    cell = ase_io.read(cell_f)
    sname = seedname(cell_f)
    num_atoms = np.size(atoms_ind)
    masses = parse_castep_masses(cell)
    cell.set_masses(masses)
    #Parse muon data
    sel = AtomSelection.from_array(
        cell, 'castep_custom_species', mu_sym)
    mu_indices = sel.indices

    if ase_phonons:
        #Calculate phonons using ASE
        evals, evecs = ase_phonon_calc(cell, dftb_phonons)
    else:
        # Parse CASTEP phonon data into casteppy object
        pd = PhononData(sname)
        # Convert frequencies back to cm-1
        pd.convert_e_units('1/cm')
        # Get phonon frequencies+modes
        evals = np.array(pd.freqs)
        evecs = np.array(pd.eigenvecs)

    # Convert frequencies to radians/second
    evals = evals*1e2*cnst.c*np.pi*2

    if method == 'wavefunction':
        maj_evecs_index = np.zeros((num_atoms, 3))
        maj_evals = np.zeros((num_atoms, 3))
        R = np.zeros((num_atoms, 3))
        maj_evecs = np.zeros((num_atoms, 3, 3))
        maj_evecs_ortho = np.zeros((num_atoms, 3, 3))

        for i, atom_ind in enumerate(atoms_ind):
            # Get major phonon modes
            maj_evecs_index[i], maj_evecs[i], maj_evecs_ortho[i] = get_major_emodes(evecs[0], atom_ind-1)
            # Get major phonon frequencies
            maj_evals[i] = np.array(evals[0][maj_evecs_index[i].astype(int)])
            # Displacements in Angstrom
            R[i] = np.sqrt(cnst.hbar/(maj_evals[i]*masses[atom_ind-1]*cnst.u))*1e10

    # Write cells with displaced atoms
    if args_w:
        if method == 'wavefunction':
            for i, atom_ind in enumerate(atoms_ind):
                try:
                    os.stat('{0}_{1}'.format(sname, atom_ind))
                except:
                    os.mkdir('{0}_{1}'.format(sname, atom_ind))
                for j, Rj in enumerate(R[i]):
                    cell.info['name'] = "{0}".format(j+1)
                    dirname = '{0}_{1}/{2}'.format(sname, atom_ind, j+1)
                    #Create linear space generator and save displaced cell files
                    lg = create_displaced_cells(cell, atom_ind-1, grid_n, 3*maj_evecs[i][j]*Rj)
                    collection = AtomsCollection(lg)
                    for atom in collection:
                        atom.set_calculator(cell.calc)
                    collection.save_tree(dirname, "cell")
                    #Copy parameter file if specified
                    if pname:
                        for k in range(grid_n):
                            shutil.copy(pname, os.path.join(dirname,
                                 '{0}_{1}/{0}_{1}.param'.format(j+1, k)))

        elif method == 'thermal':
            therm_line = np.zeros(np.size(evals[0]))
            coefficients = np.zeros(np.size(evals[0]))
            # Calculate normal mode coordinates
            for i in range(np.size(therm_line)):
                therm_line[i] = np.sqrt(1/(2*evals[0][i]))
            for iteration in range(grid_n):
                # Generate thermal line with random coefficients
                for i in range(np.size(coefficients)):
                    coefficients[i] = random.choice([-1, 1])
                therm_line = therm_line*coefficients
                # Generate inverse of the above thermal line
                therm_line_inv = therm_line*-1

                # Create folder to store displaced cell files
                dirname = '{0}_lines'.format(sname)
                try:
                    os.stat(dirname)
                except:
                    os.mkdir(dirname)
                # Set up displaced cells
                cell_thermal = cell.copy()
                cell_thermal_inv = cell.copy()
                pos = cell.get_positions()
                pos_therm = pos.copy()
                pos_therm_inv = pos.copy()

                for i in range(np.size(cell)):
                    for j in range(3, np.size(evals[0])):
                        pos_therm[i] += therm_line[j]*evecs[0][j][i].real*1e10
                        pos_therm_inv[i] += therm_line_inv[j]*evecs[0][j][i].real*1e10

                cell_thermal.set_positions(pos_therm)
                cell_thermal_inv.set_positions(pos_therm_inv)
                cell_thermal.set_calculator(cell.calc)
                cell_thermal_inv.set_calculator(cell.calc)
                ase_io.write(dirname+"/{0}_thermal_{1}.cell".format(sname, iteration), cell_thermal)
                ase_io.write(dirname+"/{0}_thermal_inv_{1}.cell".format(sname, iteration), cell_thermal_inv)

                if pname:
                    shutil.copy(pname, os.path.join(dirname,
                         "{0}_thermal_{1}.param".format(sname, iteration)))
                    shutil.copy(pname, os.path.join(dirname,
                         "{0}_thermal_inv_{1}.param".format(sname, iteration)))

    else:
        if method == 'wavefunction':
            if value_type == 'scalar':
                grid_tensors = np.zeros((np.size(cell), np.size(R[0]), grid_n, 1, 1))
            elif value_type == 'vector':
                grid_tensors = np.zeros((np.size(cell), np.size(R[0]), grid_n, 1, 3))
            elif value_type == 'matrix':
                grid_tensors = np.zeros((np.size(cell), np.size(R[0]), grid_n, 3, 3))

            # Parse tensors from appropriate files and energy from .castep files
            for i, atom_ind in enumerate(atoms_ind):
                E_table = np.zeros((np.size(R[i]), grid_n))
                dirname = '{0}_{1}'.format(sname, atom_ind)
                for j in range(np.size(R[i])):
                    for k in range(grid_n):
                        if property == 'hyperfine':
                            tensor_file = os.path.join(dirname,
                                '{0}/{0}_{1}/{0}_{1}.magres'.format(j+1, k))
                            tensors = (parse_hyperfine_magres(tensor_file)).get_array('hyperfine')
                        for l, tensor in enumerate(tensors):
                            grid_tensors[l][j][k][:][:] = tensor
                        castf = os.path.join(dirname,
                            '{0}/{0}_{1}/{0}_{1}.castep'.format(j+1, k))
                        E_table[j][k] = parse_final_energy(castf)

                symbols = cell.get_array('castep_custom_species')

                #Calculate vibrational average of property and write it out
                if weight_type == 'harmonic':
                    weighting = calc_wavefunction(R[i], grid_n, write_table = True,
                        filename = dirname+"/{0}_{1}_psi.dat".format(sname, atom_ind))
                tens_avg = weighted_tens_avg(grid_tensors, weighting)
                write_tensors(tens_avg, dirname+"/{0}_{1}_tensors.dat".format(sname, atom_ind), symbols)
                calc_harm_potential(R[i], grid_n, masses[atom_ind-1], maj_evals[i], E_table,
                    dirname+"/{0}_{1}_V.dat".format(sname, atom_ind))

                if property == 'hyperfine':
                    #Find ipso hydrogens
                    iH_indices = np.zeros(np.size(mu_indices), int)
                    for i in range(np.size(iH_indices)):
                        iH_indices[i] = find_ipso_hydrogen(mu_indices[i], cell, mu_sym)
                    #Calculate and write out hfcc for muons and ipso hydrogens
                    muon_ipso_dict = {}
                    for index in mu_indices:
                        muon_ipso_dict[index] = symbols[index]
                    for index in iH_indices:
                        muon_ipso_dict[index] = symbols[index]
                    hfine_report(R[i], grid_n, grid_tensors, tens_avg, weighting,
                    dirname+"/{0}_{1}_report.dat".format(sname, atom_ind), muon_ipso_dict)

        elif method == 'thermal':
            if value_type == 'scalar':
                tensors_avg = np.zeros((np.size(cell), 1, 1))
            elif value_type == 'vector':
                tensors_avg = np.zeros((np.size(cell), 1, 3))
            elif value_type == 'matrix':
                tensors_avg = np.zeros((np.size(cell), 3, 3))

            dirname = '{0}_lines'.format(sname)

            for iteration in range(grid_n):
                if property == 'hyperfine':
                    tensor_file = os.path.join(dirname,
                        "{0}_thermal_{1}.magres".format(sname, iteration))
                    tensor_file_inv = os.path.join(dirname,
                        "{0}_thermal_inv_{1}.magres".format(sname, iteration))
                    tensors = (parse_hyperfine_magres(tensor_file)).get_array('hyperfine')
                    tensors_inv = (parse_hyperfine_magres(tensor_file_inv)).get_array('hyperfine')
                tensors_avg += (tensors+tensors_inv)/2.0
            tensors_avg /= grid_n

            symbols = cell.get_array('castep_custom_species')
            write_tensors(tensors_avg, "{0}_tensors.dat".format(sname), symbols)

    return
