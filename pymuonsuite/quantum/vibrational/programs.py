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
from soprano.selection import AtomSelection
from soprano.utils import seedname

from pymuonsuite.io.castep import parse_castep_masses, parse_final_energy
from pymuonsuite.io.magres import parse_hyperfine_magres
from pymuonsuite.io.output import write_tensors
from pymuonsuite.quantum.vibrational.grid import create_displaced_cell
from pymuonsuite.quantum.vibrational.grid import calc_wavefunction, weighted_tens_avg
from pymuonsuite.quantum.vibrational.grid import wf_disp_generator, tl_disp_generator
from pymuonsuite.quantum.vibrational.reports import harm_potential_report, hfine_report
from pymuonsuite.quantum.vibrational.phonons import ase_phonon_calc
from pymuonsuite.quantum.vibrational.phonons import get_major_emodes
from pymuonsuite.utils import find_ipso_hydrogen
try:
    from casteppy.data.phonon import PhononData
except ImportError:
    raise ImportError("""
Can't use castep phonon interface due to casteppy not being installed.
Please download and install casteppy from Bitbucket:

HTTPS:  https://bitbucket.org/casteppy/casteppy.git
SSH:    git@bitbucket.org:casteppy/casteppy.git

and try again.""")

def vib_avg(cell_f, method, mu_sym, grid_n, property, value_type, atoms_ind=[0],
                weight_type='harmonic', pname=None, args_w=False,
                ase_phonons=False, dftb_phonons=True):
    """
    (Write mode) Given the structure and phonon modes of a molecule, write
    out a set of structure files with the atoms displaced according to the
    selected vibrational averaging method.
    (Read mode) Read in tensors calculated for each displaced structure
    created with the write mode. Perform a weighted average on these tensors
    appropriate to the mode and weighting selected. Return the vibrationally
    averaged property tensors in a file.

    | Args:
    |   cell_f (str): Path to structure file (e.g. .cell file for CASTEP)
    |   method (str): Method used to calculate thermal average. Currently
    |       accepted valus: 'wavefunction' (wavefunction sampling), 'thermal'
    |       (thermal lines)
    |   mu_sym (str): Symbol used to represent muon in structure file
    |   grid_n (int): Number of increments to make along each phonon axis
    |   property(str): Property to be calculated. Currently accepted values:
    |       "hyperfine"
    |   atoms_ind(int array): Array of indices of atoms to be displaced,
    |       counting from 0. E.g. for first 3 atoms in cell file enter [0,1,2].
    |       Enter [-1] to select all atoms.
    |   value_type(str): Is tensor being averaged a 'matrix', 'vector', or
    |       'scalar'? (e.g. the hyperfine coupling tensor is a matrix)
    |   weight_type(str): Type of weighting to be used, currently accepted
    |       values: "harmonic" (harmonic oscillator wavefunction)
    |   pname (str): Path of param file which will be copied into folders
    |       along with displaced cell files for convenience
    |   args_w (bool): Write mode if true, read mode if false
    |   ase_phonons(bool): If true, use ASE to calculate phonon modes. ASE will
    |       use the calculator of the input structure file, e.g. CASTEP for
    |       .cell files. Set dftb_phonons to True in order to use dftb+ as the
    |       calculator instead. If false, will read in CASTEP phonons.
    |   dftb_phonons(bool): Use dftb+ with ASE to calculate phonons if true.
    |       Requires ase_phonons set to true.
    |
    | Returns: Nothing
    """

    # Parse cell data
    cell = ase_io.read(cell_f)
    sname = seedname(cell_f)
    num_atoms = np.size(cell)
    symbols = cell.get_array('castep_custom_species')
    # Set correct custom species masses in cell
    masses = parse_castep_masses(cell)
    cell.set_masses(masses)
    # Parse muon data
    sel = AtomSelection.from_array(
        cell, 'castep_custom_species', mu_sym)
    mu_indices = sel.indices

    # Select all atoms if -1 input
    if atoms_ind[0] == -1:
        atoms_ind = np.arange(0, num_atoms, 1, int)
    # Thermal method requires atoms_ind = [0]
    if method == 'thermal':
        atoms_ind = np.array([0])
    num_sel_atoms = np.size(atoms_ind) #Number of atoms selected
    # Set total number of grid points
    if method == 'wavefunction':
        total_grid_n = 3*grid_n
    elif method == 'thermal':
        total_grid_n = 2*grid_n

    # Get phonons
    if ase_phonons:
        # Calculate phonons using ASE
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

    # Find 3 major modes for each atom selected and use them to calculate
    # displacement factors R for the wavefunction method
    if method == 'wavefunction':
        maj_evecs_index = np.zeros((num_sel_atoms, 3))
        maj_evecs = np.zeros((num_sel_atoms, 3, 3))
        maj_evals = np.zeros((num_sel_atoms, 3))
        R = np.zeros((num_sel_atoms, 3))

        for i, atom_ind in enumerate(atoms_ind):
            # Get major phonon modes
            maj_evecs_index[i], maj_evecs[i] = get_major_emodes(evecs[0], atom_ind)
            # Get major phonon frequencies
            maj_evals[i] = np.array(evals[0][maj_evecs_index[i].astype(int)])
            # Displacement factors in Angstrom
            R[i] = np.sqrt(cnst.hbar/(maj_evals[i]*masses[atom_ind]*cnst.u))*1e10

    # Write mode: write cells with atoms displaced
    if args_w:
        displacements = np.zeros((num_sel_atoms, total_grid_n, num_atoms, 3))

        if method == 'wavefunction':
            # For each atom selected, displace that atom but not the others
            for i, atom_ind in enumerate(atoms_ind):
                displacements[i, :, atom_ind] = wf_disp_generator(R[i], maj_evecs[i], grid_n)

        elif method == 'thermal':
            norm_coords = np.zeros(np.size(evals[0]) - 3)
            # Calculate normal mode coordinates
            for i in range(np.size(norm_coords)):
                norm_coords[i] = np.sqrt(1/(2*evals[0][i+3]))
            # Calculate displacements at this quantum point and its inverse
            for point in range(grid_n):
                point_displacements = tl_disp_generator(norm_coords, evecs[0][3:], num_atoms)
                displacements[0][point] = point_displacements
                displacements[0][point + grid_n] = -point_displacements

        # Create and write displaced cell files
        for i, atom_ind in enumerate(atoms_ind):
            # Create folder for cell files
            if method == 'wavefunction':
                dirname = '{0}_{1}_wvfn'.format(sname, atom_ind)
            elif method == 'thermal':
                dirname = '{0}_thermal'.format(sname)
            try:
                os.stat(dirname)
            except:
                os.mkdir(dirname)
            for point in range(total_grid_n):
                # Generate displaced cell
                disp_cell = create_displaced_cell(cell, displacements[i][point])
                # Write displaced cell
                ase_io.write(os.path.join(dirname,'{0}_{1}.cell'.format(sname, point)),
                                disp_cell)
                # Copy param files
                if pname:
                    shutil.copy(pname, os.path.join(dirname,
                        '{0}_{1}.param'.format(sname, point)))

    # Read mode: Read in and average tensors
    else:
        # Create appropriately sized container for reading in tensors
        if value_type == 'scalar':
            grid_tensors = np.zeros((total_grid_n, num_atoms, 1, 1))
        elif value_type == 'vector':
            grid_tensors = np.zeros((total_grid_n, num_atoms, 1, 3))
        elif value_type == 'matrix':
            grid_tensors = np.zeros((total_grid_n, num_atoms, 3, 3))

        for i, atom_ind in enumerate(atoms_ind):
            if method == 'wavefunction':
                dirname = '{0}_{1}_wvfn'.format(sname, atom_ind)
            elif method == 'thermal':
                dirname = '{0}_thermal'.format(sname)

            # Parse tensors from each grid point
            for point in range(total_grid_n):
                if property == 'hyperfine':
                    tensor_file = os.path.join(dirname,
                                    '{0}_{1}.magres'.format(sname, point))
                    magres = parse_hyperfine_magres(tensor_file)
                    grid_tensors[point] = magres.get_array('hyperfine')

            # Compute weights for each grid point
            if method == 'wavefunction':
                if weight_type == 'harmonic':
                    weighting = calc_wavefunction(R[i], grid_n, write_table = True,
                     filename = "{0}_{1}_psi.dat".format(sname, atom_ind))
            elif method == 'thermal':
                weighting = np.ones((total_grid_n)) #(uniform weighting)

            # Compute average tensors
            tens_avg = weighted_tens_avg(grid_tensors, weighting)

            # Write averaged tensors
            outfile = "{0}_{1}_tensors.dat".format(sname, atom_ind)
            write_tensors(tens_avg, outfile, symbols)

            if property == 'hyperfine':
                # Find ipso hydrogens
                iH_indices = np.zeros(np.size(mu_indices), int)
                for i in range(np.size(iH_indices)):
                    iH_indices[i] = find_ipso_hydrogen(mu_indices[i], cell, mu_sym)
                # Calculate and write out hfcc for muons and ipso hydrogens
                muon_ipso_dict = {}
                for index in mu_indices:
                    muon_ipso_dict[index] = symbols[index]
                for index in iH_indices:
                    muon_ipso_dict[index] = symbols[index]
                hfine_report(total_grid_n, grid_tensors, tens_avg, weighting,
                "{0}_{1}_report.dat".format(sname, atom_ind), muon_ipso_dict)

            if method == 'wavefunction' and weight_type == 'harmonic':
                # Grab CASTEP final energies
                E_table = np.zeros((np.size(R[i]), grid_n))
                for j in range(np.size(E_table, 0)):
                    for k in range(np.size(E_table, 1)):
                        castf = os.path.join(dirname, "{0}_{1}.castep".format(sname, k+j*grid_n))
                        E_table[j][k] = parse_final_energy(castf)
                # Write harmonic potential report
                harm_potential_report(R[i], grid_n, masses[atom_ind], maj_evals[i],
                     E_table, "{0}_{1}_V.dat".format(sname, atom_ind))

    return
