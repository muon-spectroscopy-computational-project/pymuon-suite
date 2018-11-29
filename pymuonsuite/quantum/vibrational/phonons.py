"""Phonons extracted from CASTEP results

Requites casteppy installed

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
from ase.calculators.dftb import Dftb
from ase.dft import kpoints
from ase.phonons import Phonons
from soprano.collection import AtomsCollection
from soprano.collection.generate import linspaceGen
from soprano.utils import seedname

from pymuonsuite.io.castep import parse_final_energy, parse_castep_muon
from pymuonsuite.io.magres import parse_hyperfine_magres
from pymuonsuite.io.output import write_tensors
from pymuonsuite.quantum.grid import calc_wavefunction, weighted_tens_avg
from pymuonsuite.utils import find_ipso_hydrogen, create_displaced_cells
try:
    from casteppy.data.phonon import PhononData
except ImportError:
    raise ImportError("""
Can't use castep phonon interface due to casteppy not being installed.
Please download and install casteppy from Bitbucket:

HTTPS:  https://bitbucket.org/casteppy/casteppy.git
SSH:    git@bitbucket.org:casteppy/casteppy.git

and try again.""")

def ase_phonon_calc(cell, dftb_phonons):
    """Calculate phonon modes of a molecule using ASE. If dftb_phonons is true,
    DFTB+ will be used as the calculator. Otherwise, the input cell's calculator
    will be used. A report of the phonon modes will be written to a file and
    arrays of the eigenvectors and eigenvalues returned.

    | Args:
    |   cell(ASE Atoms object): Atoms object with geometry to calculate modes
    |   for.
    |   dftb_phonons(bool): If True, use DFTB+ to calculate. If False, use
    |   cell's default calculator.
    | Returns:
    |   evals(float[k-points][modes]): Eigenvalues of phonon modes
    |   evecs(float[k-points][modes][ions][3]): Eigenvectors of phonon modes
    """
    if dftb_phonons:
        phonon_calc = Dftb(kpts=[1,1,1])
    else:
        phonon_calc = cell.get_calculator()
    ph = Phonons(cell, phonon_calc)
    ph.run()
    ph.read(acoustic=True)
    path = kpoints.monkhorst_pack((1,1,1))
    evals, evecs = ph.band_structure(path, True)
    evals *= 8065.5 #Convert from eV to cm-1

    #Write phonon report
    filename = "ase_phonons.dat"
    phonfile = open(filename, 'a')
    print("Writing phonon report in location: ", filename)
    phonfile.write("Eigenvalues\n")
    for i, kpt in enumerate(evals):
        phonfile.write("Mode Frequency(cm-1) k-point = {0}\n".format(i))
        for j, value in enumerate(kpt):
            phonfile.write("{0} \t{1}\n".format(j, value))
    phonfile.write("Eigenvectors\n")
    phonfile.write("Mode Ion Vector\n")
    for i, mode in enumerate(evecs[0]):
        for j, ion in enumerate(mode):
            phonfile.write("{0} {1} \t{2}\n".format(i, j, ion))

    return evals, evecs

def calc_harm_potential(R, grid_n, mass, freqs, E_table, sname):
    """
    Calculate the harmonic potential at all displacements on the grid for an
    atom and write out to file in a format that can be plotted.

    | Args:
    |   R(Numpy float array, shape:(axes)): Displacement amplitude along each
    |       axis
    |   grid_n(int): Number of grid points along each axis
    |   mass(float): Mass of atom
    |   freqs(Numpy float array, shape:(axes)): Frequencies of harmonic
    |       oscillator along each axis
    |   E_table(Numpy float array, shape:(np.size(R), grid_n)): Table of CASTEP
    |       final system energies.
    |   sname(str): Seedname to be used for filename (i.e. filename will be
    |       sname_V.dat
    |
    | Returns: Nothing
    """
    R_axes = np.array([np.linspace(-3*Ri, 3*Ri, grid_n)
                       for Ri in R])
    # Now the potential, measured vs. theoretical
    harm_K = mass*freqs**2
    harm_V = (0.5*harm_K[:, None]*(R_axes*1e-10)**2)/cnst.electron_volt
    # Normalise E_table
    if E_table.shape[1] % 2 == 1:
        E_table -= (E_table[:, E_table.shape[1]//2])[:, None]
    else:
        E_table -= (E_table[:, E_table.shape[1]//2] +
                    E_table[:, E_table.shape[1]//2-1])[:, None]/2.0
    all_table = np.concatenate((R_axes, harm_V, E_table), axis=0)
    np.savetxt(sname + '_V.dat', all_table.T)

def get_major_emodes(evecs, i):
    """Find the normalized phonon modes of the atom at index i

    | Args:
    |   evecs (Numpy float array, shape: (num_modes, num_ions, 3)):
    |                                   Eigenvectors of phonon modes of molecule
    |   i (int): Index of atom in position array
    |
    | Returns:
    |   major_evecs_i (int[3]): Indices of atom's phonon eigenvectors in evecs
    |   major_evecs (float[3]): Normalized eigenvectors of atom's phonon modes
    |   major_evecs_ortho (float[3]): Orthogonalised phonon modes
    """
    # First, find the eigenmodes whose amplitude is greater for ion i
    evecs_amp = np.linalg.norm(evecs, axis=-1)
    ipr = evecs_amp**4/np.sum(evecs_amp**2, axis=-1)[:, None]**2
    evecs_order = np.argsort(ipr[:, i])

    # How many?
    major_evecs_i = evecs_order[-3:]
    major_evecs = evecs[major_evecs_i, i]
    major_evecs_ortho = np.linalg.qr(major_evecs.T)[0].T

    major_evecs = major_evecs/np.linalg.norm(major_evecs, axis=-1, keepdims=True)
    major_evecs = np.real(major_evecs)

    return major_evecs_i, major_evecs, major_evecs_ortho

def phonon_hfcc(cell_f, mu_sym, grid_n, calc='castep', pname=None,
                ignore_ipsoH=False, solver=False, args_w=False,
                ase_phonons=False, dftb_phonons=False):
    """
    Given a file containing phonon modes of a muoniated molecule, either write
    out a set of structure files with the muon progressively displaced in
    grid_n increments along the axes of the phonon modes, or read in hyperfine
    coupling values from a set of .magres files with such a set of muon
    displacements and average them to give an estimate of the actual hfcc
    accounting for nuclear quantum effects.

    | Args:
    |   cell_f (str): Path to structure file (e.g. .cell file for CASTEP)
    |   mu_sym (str): Symbol used to represent muon in structure file
    |   grid_n (int): Number of increments to make along each phonon axis
    |   calc (str): Calculator used (e.g. CASTEP)
    |   pname (str): Path of param file which will be copied into folders
    |       along with displaced cell files for convenience
    |   ignore_ipsoH (bool): If true, ignore ipso hydrogen calculations
    |   solver (bool): If true, use qlab to numerically solve the schroedinger
    |       equation
    |   args_w (bool): Write files if true, parse if false
    |   ase_phonons(bool): If True, use ASE to calculate phonon modes. ASE will
    |       use the calculator of the input cell, e.g. CASTEP for .cell files. Set
    |       dftb_phonons to True in order to use dftb+ as the calculator instead.
    |       If False, will read in CASTEP phonons.
    |   dftb_phonons(bool): Use dftb+ to calculate phonons if true. Requires
    |       ase_phonons set to true
    |
    | Returns: Nothing
    """
    cell = ase_io.read(cell_f)
    sname = seedname(cell_f)
    #Parse muon data using appropriate parser for calculator
    if (calc.strip().lower() in 'castep'):
        mu_index, ipso_H_index, mu_mass = parse_castep_muon(cell, mu_sym,
                                                            ignore_ipsoH)
    else:
        raise RuntimeError("Invalid calculator entered ('{0}').".format(calc))

    if ase_phonons:
        #Calculate phonons using ASE
        masses = cell.get_masses()
        masses[-1] = mu_mass/cnst.u
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
        # Parse hyperfine values from .magres files and energy from .castep
        # files
        E_table = []
        hfine_table = ipso_hfine_table = np.zeros((np.size(R), grid_n))
        num_species = np.size(cell.get_array('castep_custom_species'))
        all_hfine_tensors = np.zeros((num_species, np.size(R), grid_n, 3, 3))
        for i, Ri in enumerate(R):
            E_table.append([])
            dirname = '{0}_{1}'.format(sname, i+1)
            for j in range(grid_n):
                mfile = os.path.join(dirname,
                    '{0}_{1}_{2}/{0}_{1}_{2}.magres'.format(sname, i+1, j))
                mgr = parse_hyperfine_magres(mfile)
                hfine_table[i][j] = np.trace(
                    mgr.get_array('hyperfine')[mu_index])/3.0
                if not ignore_ipsoH:
                    ipso_hfine_table[i][j] = np.trace(
                        mgr.get_array('hyperfine')[ipso_H_index])/3.0
                else:
                    ipso_hfine_table = None
                for k, tensor in enumerate(mgr.get_array('hyperfine')):
                    all_hfine_tensors[k][i][j][:][:] = tensor
                castf = os.path.join(dirname,
                    '{0}_{1}_{2}/{0}_{1}_{2}.castep'.format(sname, i+1, j))
                E_table[-1].append(parse_final_energy(castf))

        E_table = np.array(E_table)
        if (hfine_table.shape != (3, grid_n) or
                E_table.shape != (3, grid_n)):
            raise RuntimeError("Incomplete or absent magres or castep data")

        symbols = cell.get_array('castep_custom_species')

        r2psi2 = calc_wavefunction(R, grid_n, E_table = E_table,
            write_table = True, value_table = hfine_table, sname = sname)

        hfine_tens_avg = weighted_tens_avg(all_hfine_tensors, r2psi2)
        write_tensors(hfine_tens_avg, sname, symbols)

        calc_harm_potential(R, grid_n, mu_mass, mu_evals, E_table, sname)

    return
