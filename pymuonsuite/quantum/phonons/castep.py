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
from soprano.selection import AtomSelection
from soprano.utils import seedname

from pymuonsuite.io.castep import parse_final_energy, parse_phonon_file
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
    a_i varying between -evecs*3*R and +evecs*3*R with grid_n increments

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

def castep_muon_modes(cell_f, mu_sym, ignore_ipsoH):
    """Return data about muon index, mass, phonon modes and eigenvalues and
       optionally the ipso hydrogen's index from a castep .cell and .phonon
       file.

    | Args:
    |   cell_f (str): Path of cell file
    |   mu_sym (str): Symbol used to represent muon
    |   ignore_ipsoH (bool): If true, do not find ipso hydrogen index
    |
    | Returns:
    |   cell (ASE Atoms object): ASE structure data
    |   sname (str): Cell file name minus file extension
    |   mu_index (int): Index of muon in cell file
    |   ipso_H_index (int): Index of ipso hydrogen in cell file
    |   mu_mass (float): Mass of muon
    |   em_i (int[3]): Indices of eigenvectors in evec array
    |   em (float[3]): Eigenvectors of muon phonon modes
    |   em_o (float[3]):
    |   evals (float[3]): Eigenvalues of muon phonon modes
    """
    # Get seedname
    sname = seedname(cell_f)
    # Parse phonon data into object
    pd = PhononData(sname)
    # Convert frequencies back to cm-1
    pd.convert_e_units('1/cm')
    # Create eigenvector array that is formatted to work with get_major_emodes.
    evecs = np.zeros((pd.n_qpts, pd.n_branches, pd.n_ions, 3),
                     dtype='complex128')
    for i in range(pd.n_qpts):
        for j in range(pd.n_branches):
            for k in range(pd.n_ions):
                evecs[i][j][k][:] = pd.eigenvecs[i][j*pd.n_ions+k][:]

    # Read in cell file
    cell = ase_io.read(sname + '.cell')
    cell.info['name'] = sname
    # Find muon index in structure array
    sel = AtomSelection.from_array(
        cell, 'castep_custom_species', mu_sym)
    mu_index = sel.indices[0]
    # Get muon mass
    mu_mass = float(cell.calc.cell.species_mass.value.split()[2])
    mu_mass = mu_mass*cnst.u  # Convert to kg

    # Get muon phonon modes
    em_i, em, em_o = get_major_emodes(evecs[0], mu_index)
    em = np.real(em)

    # Find ipso hydrogen location
    if not ignore_ipsoH:
        ipso_H_index = find_ipso_hydrogen(mu_index, cell, mu_sym)
    else:
        ipso_H_index = None

    # Get muon phonon frequencies and convert to radians/second
    evals = np.array(pd.freqs[0][em_i]*1e2*cnst.c*np.pi*2)

    return cell, sname, mu_index, ipso_H_index, mu_mass, em_i, em, em_o, evals

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
    #Parse muon data using appropriate parser for calculator
    if (calc.strip().lower() in 'castep'):
        cell, sname, mu_index, ipso_H_index, mu_mass, em_i, em, em_o, evals = \
        castep_muon_modes(cell_f, mu_sym, ignore_ipsoH)
    else:
        raise RuntimeError("Invalid calculator entered ('{0}').".format(calc))

    # Displacement in Angstrom
    R = np.sqrt(cnst.hbar/(evals*mu_mass))*1e10

    # Write cells with displaced muon
    if args_w:
        for i, Ri in enumerate(R):
            cell.info['name'] = sname + '_' + str(i+1)
            dirname = '{0}_{1}'.format(sname, i+1)
            lg = create_displaced_cells(cell, mu_index, grid_n, 3*em[i]*Ri)
            collection = AtomsCollection(lg)
            collection.save_tree(dirname, "cell")
            #Copy paramater file if specified
            if pname:
                for j in range(grid_n):
                    shutil.copy(pname, os.path.join(dirname,
                         '{0}_{1}_{2}/{0}_{1}_{2}.param'.format(sname, i+1, j)))

    else:
        # Parse hyperfine values from .magres files and energy from .castep
        # files
        E_table = []
        hfine_table = np.zeros((np.size(R), grid_n))
        ipso_hfine_table = np.zeros((np.size(R), grid_n))
        num_species = np.size(cell.get_array('castep_custom_species'))
        all_hfine_tensors = np.zeros(
            (num_species, np.size(R), grid_n, 3, 3))
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
                            mu_mass, evals, E_table, sname)

    return
