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

def vib_avg_muon(cell_f, mu_sym, grid_n, property, value_type, weight_type,
                pname=None, ignore_ipsoH=False, solver=False, args_w=False,
                ase_phonons=False, dftb_phonons=True):
    """
    (Write mode) Given a file containing phonon modes of a muonated molecule,
    write out a set of structure files with the muon progressively displaced in
    grid_n increments along each axis of the phonon modes, creating a grid.
    (Read mode) Read in coupling values calculated at each point of a grid
    created using this function's write mode and average them to give an estimate
    of the actual coupling values accounting for nuclear quantum effects.

    | Args:
    |   cell_f (str): Path to structure file (e.g. .cell file for CASTEP)
    |   mu_sym (str): Symbol used to represent muon in structure file
    |   grid_n (int): Number of increments to make along each phonon axis
    |   property(str): Property to be calculated. Currently accepted values:
    |       "hyperfine" (hyperfine tensors),
    |   value_type(str): Is value being calculated a 'matrix', 'vector', or
    |       'scalar'? (e.g. hyperfine tensor is a matrix)
    |   weight_type(str): Type of weighting to be used, currently accepted
    |       values: "harmonic" (harmonic oscillator wavefunction)
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
            #Create linear space generator and save displaced cell files
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

def vib_avg_all(cell_f, mu_sym, grid_n, property, value_type, weight_type,
                pname=None, ignore_ipsoH=False, solver=False, args_w=False,
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
    |   property(str): Property to be calculated. Currently accepted values:
    |       "hyperfine" (hyperfine tensors),
    |   value_type(str): Is value being calculated a 'matrix', 'vector', or
    |       'scalar'? (e.g. hyperfine tensor is a matrix)
    |   weight_type(str): Type of weighting to be used, currently accepted
    |       values: "harmonic" (harmonic oscillator wavefunction)
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
    num_atoms = np.size(cell)
    masses = parse_castep_masses(cell)
    cell.set_masses(masses)
    #Parse muon data
    sel = AtomSelection.from_array(
        cell, 'castep_custom_species', mu_sym)
    mu_indices = sel.indices
    # Find ipso hydrogen location(s)
    if not ignore_ipsoH:
        iH_indices = np.zeros(np.size(mu_indices), int)
        for i, index in enumerate(iH_indices):
            index = find_ipso_hydrogen(mu_indices[i], cell, mu_sym)
    else:
        iH_indices = None

    if ase_phonons:
        #Calculate phonons using ASE
        evals, evecs = ase_phonon_calc(cell, dftb_phonons)
    else:
        # Parse CASTEP phonon data into casteppy object
        pd = PhononData(sname)
        # Convert frequencies back to cm-1
        pd.convert_e_units('1/cm')
        # Get phonon frequencies+modes
        evals = pd.freqs
        evecs = pd.eigenvecs

    maj_evecs_index = np.zeros((num_atoms, 3))
    maj_evals = np.zeros((num_atoms, 3))
    R = np.zeros((num_atoms, 3))
    maj_evecs = np.zeros((num_atoms, 3, 3))
    maj_evecs_ortho = np.zeros((num_atoms, 3, 3))

    for index in range(num_atoms):
        # Get major phonon modes
        maj_evecs_index[index], maj_evecs[index], maj_evecs_ortho[index] = get_major_emodes(evecs[0], index)
        # Get major phonon frequencies and convert to radians/second
        maj_evals[index] = np.array(evals[0][maj_evecs_index[index].astype(int)]*1e2*cnst.c*np.pi*2)
        # Displacements in Angstrom
        R[index] = np.sqrt(cnst.hbar/(maj_evals[index]*masses[index]*cnst.u))*1e10

    # Write cells with displaced muon
    if args_w:
        for i in range(num_atoms):
            try:
                os.stat('{0}_{1}'.format(sname, i+1))
            except:
                os.mkdir('{0}_{1}'.format(sname, i+1))
            for j, Rj in enumerate(R[i]):
                cell.info['name'] = "{0}".format(j+1)
                dirname = '{0}_{1}/{2}'.format(sname, i+1, j+1)
                #Create linear space generator and save displaced cell files
                lg = create_displaced_cells(cell, i, grid_n, 3*maj_evecs[i][j]*Rj)
                collection = AtomsCollection(lg)
                for atom in collection:
                    atom.set_calculator(cell.calc)
                collection.save_tree(dirname, "cell")
                #Copy parameter file if specified
                if pname:
                    for k in range(grid_n):
                        shutil.copy(pname, os.path.join(dirname,
                             '{0}_{1}/{0}_{1}.param'.format(j+1, k)))

    else:
        E_table = np.zeros((np.size(R[0]), grid_n))
        if value_type == 'scalar':
            grid_tensors = np.zeros((num_atoms, np.size(R[0]), grid_n, 1, 1))
        elif value_type == 'vector':
            grid_tensors = np.zeros((num_atoms, np.size(R[0]), grid_n, 1, 3))
        elif value_type == 'matrix':
            grid_tensors = np.zeros((num_atoms, np.size(R[0]), grid_n, 3, 3))

        # Parse tensors from appropriate files and energy from .castep files
        for i in range(np.size(cell)):
            dirname = '{0}_{1}'.format(sname, i+1)
            for j in range(np.size(R[0])):
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
                    filename = dirname+"/{0}_{1}_psi.dat".format(sname, i+1))
            tens_avg = weighted_tens_avg(grid_tensors, weighting)
            write_tensors(tens_avg, dirname+"/{0}_{1}_tensors.dat".format(sname, i+1), symbols)
            calc_harm_potential(R[i], grid_n, masses[i], maj_evals[i], E_table,
                dirname+"/{0}_{1}_V.dat".format(sname, i+1))

            if property == 'hyperfine':
                muon_ipso_dict = {}
                for index in mu_indices:
                    muon_ipso_dict[index] = symbols[index]
                for index in iH_indices:
                    muon_ipso_dict[index] = symbols[index]
                hfine_report(R[i], grid_n, grid_tensors, tens_avg, weighting,
                dirname+"/{0}_{1}_report.dat".format(sname, i+1), muon_ipso_dict)

    return
