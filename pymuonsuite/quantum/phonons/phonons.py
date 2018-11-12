"""Phonons extracted from CASTEP results

Requites casteppy installed
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
from soprano.collection.generate import linspaceGen
from soprano.utils import seedname

from pymuonsuite.io.castep import parse_final_energy, parse_castep_muon
from pymuonsuite.io.magres import parse_hyperfine_magres
from pymuonsuite.quantum.grid import calc_wavefunction, avg_hfine_tensor
from pymuonsuite.quantum.grid import write_tensors, calc_harm_potential
from pymuonsuite.utils import find_ipso_hydrogen
from pymuonsuite.quantum.phonons.utils import get_major_emodes
try:
    from casteppy.data.phonon import PhononData
except ImportError:
    raise ImportError("""
Can't use castep phonon interface due to casteppy not being installed.
Please download and install casteppy from Bitbucket:

HTTPS:  https://bitbucket.org/casteppy/casteppy.git
SSH:    git@bitbucket.org:casteppy/casteppy.git

and try again.""")



def create_displaced_cells(cell, a_i, grid_n, disp):
    """Create a range ASE Atoms objects with the displacement of atom at index
    a_i varying between -disp and +disp with grid_n increments

    | Args:
    |   cell (ASE Atoms object): Object containing atom to be displaced
    |   a_i (int): Index of atom to be displaced
    |   grid_n (int): Number of increments/objects to create
    |   disp (float): Maximum displacement from original position
    |
    | Returns:
    |   lg(Soprano linspaceGen object): Generator of displaced cells
    """
    pos = cell.get_positions()
    cell_L = cell.copy()
    pos_L = pos.copy()
    pos_L[a_i] -= disp
    cell_L.set_positions(pos_L)
    cell_R = cell.copy()
    pos_R = pos.copy()
    pos_R[a_i] += disp
    cell_R.set_positions(pos_R)
    lg = linspaceGen(
        cell_L, cell_R, steps=grid_n, periodic=True)
    return lg

def phonon_hfcc(cell_f, mu_sym, grid_n, calc='castep', pname=None,
                ignore_ipsoH=False, save_tens=False, solver=False, args_w=False):
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
    |                along with displaced cell files for convenience
    |   ignore_ipsoH (bool): If true, ignore ipso hydrogen calculations
    |   save_tens (bool): If true, save full hyperfine tensors for all atoms
    |   solver (bool): If true, use qlab to numerically solve the schroedinger
    |                  equation
    |   args_w (bool): Write files if true, parse if false
    |
    | Returns: Nothing
    """
    cell = ase_io.read(cell_f)
    sname = seedname(cell_f)
    #Parse muon data using appropriate parser for calculator
    if (calc.strip().lower() in 'castep'):
        mu_index, ipso_H_index, mu_mass = parse_castep_muon(cell, mu_sym,
                                                            ignore_ipsoH)
        # Parse phonon data into object
        pd = PhononData(sname)
        # Convert frequencies back to cm-1
        pd.convert_e_units('1/cm')
        # Get phonon frequencies
        evals = pd.freqs
        evecs = pd.eigenvecs
    else:
        raise RuntimeError("Invalid calculator entered ('{0}').".format(calc))

    # Get muon phonon modes
    mu_evecs_index, mu_evecs, mu_evecs_ortho = get_major_emodes(evecs[0], mu_index)
    mu_evecs = np.real(mu_evecs)
    #Get muon phonon frequencies and convert to radians/second
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

        r2psi2 = calc_wavefunction(R, grid_n, mu_mass, E_table,
                                   hfine_table, sname,
                                   solver, True)
        D1, D2, ipso_D1, ipso_D2 = avg_hfine_tensor(r2psi2, hfine_table,
                                                    all_hfine_tensors[
                                                        mu_index],
                                                    ignore_ipsoH,
                                                    ipso_hfine_table,
                                                    all_hfine_tensors[
                                                        ipso_H_index],
                                                    sname)
        if (save_tens):
            write_tensors(sname, all_hfine_tensors, r2psi2, symbols)
        calc_harm_potential(R, grid_n,
                            mu_mass, mu_evals, E_table, sname)

    return
