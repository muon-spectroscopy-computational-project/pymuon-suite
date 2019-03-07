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
from soprano.selection import AtomSelection
from soprano.utils import seedname

from pymuonsuite.io.castep import parse_castep_masses, parse_final_energy
from pymuonsuite.io.castep import parse_castep_bands
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


def vib_avg(cell_f, method, mu_sym, grid_n, property, selection=[0],
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
    |   grid_n (int): Number of increments to make along each phonon axis for
    |       wavefunction method, or pairs of thermal lines to generate.
    |   property(str): Property to be calculated. Currently accepted values:
    |       "hyperfine", "bandstructure"
    |   selection(int array): Array of muons to be displaced. Counting muons
    |       from 0 in the order they are in in the cell file. E.g. to select
    |       the 1st and 3rd muon in the atom positions list, enter [0, 2].
    |   weight_type(str): Type of weighting to be used, currently accepted
    |       values: "harmonic" (harmonic oscillator wavefunction)
    |   pname (str): Path of param file which will be copied into folders
    |       along with displaced cell files for convenience
    |   args_w (bool): Write mode if true, read mode if false
    |   ase_phonons(bool): If true, use ASE and DFTB+ to calculate phonon modes.
    |       If false, will read in CASTEP phonons.
    |
    | Returns: Nothing
    """

    # Parse cell data
    cell = ase_io.read(cell_f)
    sname = seedname(cell_f)
    num_atoms = np.size(cell)
    try:
        symbols = cell.get_array('castep_custom_species')
        # Set correct custom species masses in ASE cell
        masses = parse_castep_masses(cell)
        cell.set_masses(masses)
        # Find muon locations
        mu_indices = AtomSelection.from_array(
            cell, 'castep_custom_species', mu_sym).indices
    except:
        # In case no custom species used
        print("WARNING: ASE has detected no custom species in the input file.")
        symbols = cell.get_chemical_symbols()
        masses = cell.get_masses()
        mu_indices = [0]

    # Create array of selected muon indices
    if method == 'wavefunction':
        if selection[0] == -1:
            mu_sel = mu_indices
        else:
            mu_sel = []
            try:
                for i in selection:
                    mu_sel.append(mu_indices[i])
            except IndexError:
                print("""IndexError: Muon selection '{0}' out of bounds
                       with number of muons '{1}'""".format(i, len(mu_indices)))
                exit()
    else:
        mu_sel = [0]
    num_sel_mu = len(mu_sel)

    # Set total number of grid points
    if method == 'wavefunction':
        total_grid_n = 3*grid_n
    elif method == 'thermal':
        total_grid_n = 2*grid_n

    # Get phonons
    if ase_phonons:
        # Calculate phonons using ASE
        evals, evecs = ase_phonon_calc(cell)
        ortho = True
    else:
        # Parse CASTEP phonon data into casteppy object
        pd = PhononData(sname)
        # Convert frequencies back to cm-1
        pd.convert_e_units('1/cm')
        # Get phonon frequencies+modes
        evals = np.array(pd.freqs)
        evecs = np.array(pd.eigenvecs)
        ortho = False

    # Convert frequencies to radians/second
    evals = evals*1e2*cnst.c*np.pi*2

    # Find 3 major modes for each atom selected and use them to calculate
    # displacement factors R for the wavefunction method
    if method == 'wavefunction':
        masses = cell.get_masses()
        maj_evecs_index = np.zeros((num_sel_mu, 3))
        maj_evecs = np.zeros((num_sel_mu, 3, 3))
        maj_evals = np.zeros((num_sel_mu, 3))
        R = np.zeros((num_sel_mu, 3))

        for i, mu_ind in enumerate(mu_sel):
            # Get major phonon modes
            maj_evecs_index[i], maj_evecs[i] = get_major_emodes(evecs[0], masses,
                                                                mu_ind,
                                                                ortho=ortho)
            # Get major phonon frequencies
            maj_evals[i] = np.array(evals[0][maj_evecs_index[i].astype(int)])
            # Displacement factors in Angstrom
            R[i] = np.sqrt(cnst.hbar/(maj_evals[i]*masses[mu_ind]*cnst.u))*1e10

    # Write mode: write cells with atoms displaced
    if args_w:
        displacements = np.zeros((num_sel_mu, total_grid_n, num_atoms, 3))

        # Calculate displacements of muon along 3 major axes
        if method == 'wavefunction':
            # For each atom selected, displace that atom but not the others
            for i, mu_ind in enumerate(mu_sel):
                displacements[i, :, mu_ind] = wf_disp_generator(
                    R[i], maj_evecs[i], grid_n)

        # Calculate displacements for all atoms according to thermal lines
        elif method == 'thermal':
            num_modes = np.size(evals[0]) - 3
            norm_coords = np.zeros((num_atoms, num_modes))
            # Calculate normal mode coordinates
            for i in range(num_modes):
                norm_coords[:, i] = np.sqrt(cnst.hbar/(2*evals[0][i+3]))
            norm_coords /= np.sqrt(masses[:, None]*cnst.u)
            # Calculate displacements at this quantum point and its inverse
            for point in range(grid_n):
                point_displacements = tl_disp_generator(
                    norm_coords, evecs[0][3:], num_atoms)
                displacements[0][point] = point_displacements
                displacements[0][point + grid_n] = -point_displacements

        # Create and write displaced cell files
        for i, mu_ind in enumerate(mu_sel):
            # Create folder for cell files
            if method == 'wavefunction':
                dirname = '{0}_{1}_wvfn'.format(sname, mu_ind)
            elif method == 'thermal':
                dirname = '{0}_thermal'.format(sname)
            try:
                os.stat(dirname)
            except:
                os.mkdir(dirname)
            for point in range(total_grid_n):
                # Generate displaced cell
                disp_cell = create_displaced_cell(
                    cell, displacements[i][point])
                # Write displaced cell
                outfile = os.path.join(
                    dirname, '{0}_{1}.cell'.format(sname, point))
                ase_io.write(outfile, disp_cell)
                # Copy param files
                if pname:
                    shutil.copy(pname, os.path.join(dirname,
                                                    '{0}_{1}.param'.format(sname, point)))

    # Read mode: Read in and average tensors
    else:
        for i, mu_ind in enumerate(mu_sel):
            if method == 'wavefunction':
                dirname = '{0}_{1}_wvfn'.format(sname, mu_ind)
            elif method == 'thermal':
                dirname = '{0}_thermal'.format(sname)

            # Create appropriately sized container for reading in values
            if property == 'hyperfine':
                grid_tensors = np.zeros((total_grid_n, num_atoms, 3, 3))
            elif property == 'bandstructure':
                test_path = os.path.join(dirname, "{0}_0.bands".format(sname))
                num_bs_kpt, num_bs_evals = parse_castep_bands(test_path, True)
                grid_tensors = np.zeros((total_grid_n, 1, num_bs_kpt,
                                         num_bs_evals))

            # Parse tensors from each grid point
            for point in range(total_grid_n):
                if property == 'hyperfine':
                    tensor_file = os.path.join(dirname,
                                               '{0}_{1}.magres'.format(sname, point))
                    magres = parse_hyperfine_magres(tensor_file)
                    grid_tensors[point] = magres.get_array('hyperfine')
                elif property == 'bandstructure':
                    tensor_file = os.path.join(dirname,
                                               '{0}_{1}.bands'.format(sname, point))
                    grid_tensors[point] = parse_castep_bands(tensor_file)

            # Compute weights for each grid point
            if method == 'wavefunction':
                if weight_type == 'harmonic':
                    outfile = dirname + "_psi.dat"
                    weighting = calc_wavefunction(R[i], grid_n, True, outfile)
            elif method == 'thermal':
                weighting = np.ones((total_grid_n))  # (uniform weighting)

            # Compute average tensors
            tens_avg = weighted_tens_avg(grid_tensors, weighting)

            # Write averaged tensors
            outfile = dirname + "_tensors.dat"
            write_tensors(tens_avg, outfile, symbols)

            if property == 'hyperfine':
                # Find ipso hydrogens
                iH_indices = np.zeros(np.size(mu_indices), int)
                for i in range(np.size(iH_indices)):
                    iH_indices[i] = find_ipso_hydrogen(
                        mu_indices[i], cell, mu_sym)
                # Calculate and write out hfcc for muons and ipso hydrogens
                muon_ipso_dict = {}
                for index in mu_indices:
                    muon_ipso_dict[index] = symbols[index]
                for index in iH_indices:
                    muon_ipso_dict[index] = symbols[index]
                outfile = dirname + "_report.dat"
                hfine_report(total_grid_n, grid_tensors, tens_avg, weighting,
                             outfile, muon_ipso_dict)

            if method == 'wavefunction' and weight_type == 'harmonic':
                # Grab CASTEP final energies
                E_table = np.zeros((np.size(R[i]), grid_n))
                for j in range(np.size(E_table, 0)):
                    for k in range(np.size(E_table, 1)):
                        castf = os.path.join(dirname,
                                             "{0}_{1}.castep".format(sname, k+j*grid_n))
                        E_table[j][k] = parse_final_energy(castf)
                # Write harmonic potential report
                outfile = dirname + "_V.dat"
                harm_potential_report(R[i], grid_n, masses[mu_ind],
                                      maj_evals[i], E_table, outfile)

    return
