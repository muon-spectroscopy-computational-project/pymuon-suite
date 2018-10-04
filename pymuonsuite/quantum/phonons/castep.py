"""Phonons extracted from CASTEP results

Requites casteppy installed
"""

# Python 2-to-3 compatibility code
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
from ase import io as ase_io
from soprano.selection import AtomSelection
from pymuonsuite.utils import find_ipso_hydrogen
from pymuonsuite.quantum.phonons.utils import get_major_emodes
try:
    from casteppy.data.phonon import PhononData
except ImportError:
    raise ImportError("""
Can't use castep phonon interface due to casteppy not being installed.
Please download and install casteppy from GitHub:

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

def write_displaced_cells(cell, sname, pname, lg, i):
    """
    Write out all modified cells in lg using seedname "sname". Also copy
    param file at "pname" if one provided.

    | Args:
    |   cell (ASE Atoms object): Seed cell file, used to set appropriate
    |                           calculator
    |   sname (str): Seedname of cell file e.g. seedname.cell
    |   pname (str): Path of param file to be copied
    |   lg (Soprano linspaceGen object): Generator containing modified cells
    |   i (int): Numerical suffix for cell file seedname
    |
    | Returns: Nothing
    """
    dirname = '{0}_{1}'.format(sname, i+1)
    print("Creating folder", dirname)
    try:
        os.mkdir(dirname)
    except OSError:
        # Folder already exists
        pass

    for j, c in enumerate(lg):
        c.set_calculator(cell.calc)
        ase_io.write(os.path.join(dirname,
                                  '{0}_{1}_{2}.cell'.format(sname, i+1, j+1)), c)
        # If present, copy param file!
        try:
            shutil.copy(pname, os.path.join(dirname,
                                            '{0}_{1}_{2}.param'.format(sname, i+1, j+1)))
        except IOError:
            pass
    return

def phonon_hfcc(params, args_write):
    """
    Given a file containing phonon modes of a muoniated molecule, either write
    out a set of structure files with the muon progressively displaced in
    grid_n increments along the axes of the phonon modes, or read in hyperfine
    coupling values from a set of .magres files with such a set of muon
    displacements and average them to give an estimate of the actual hfcc
    accounting for nuclear quantum effects.

    | Args:
    |   params (dict): Dictionary of parameters parsed using PhononHfccSchema
    |   args_write (bool): Write files if true, parse if false
    |
    | Returns: Nothing
    """
    # Get seedname
    sname = seedname(params['cell_file'])
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
    # Find muon index in structure array
    sel = AtomSelection.from_array(
        cell, 'castep_custom_species', params['muon_symbol'])
    mu_index = sel.indices[0]
    # Get muon mass
    mu_mass = float(cell.calc.cell.species_mass.value.split()[2])
    mu_mass = mu_mass*cnst.u  # Convert to kg

    # Get muon phonon modes
    em_i, em, em_o = get_major_emodes(evecs[0], mu_index)
    em = np.real(em)

    # Find ipso hydrogen location and phonon modes
    if not params['ignore_ipsoH']:
        ipso_H_index = find_ipso_hydrogen(
            mu_index, cell, params['muon_symbol'])
        em_i_H, em_H, em_o_H = get_major_emodes(evecs[0], ipso_H_index)
        em_H = np.real(em_H)

    # Get muon phonon frequencies and convert to radians/second
    evals = np.array(pd.freqs[0][em_i]*1e2*cnst.c*np.pi*2)
    # Displacement in Angstrom
    R = np.sqrt(cnst.hbar/(evals*mu_mass))*1e10

    # Write cells with displaced muon
    if args_write:
        pname = sname + '.param'
        if not os.path.isfile(pname):
            print("WARNING - no .param file was found")
        for i, Ri in enumerate(R):
            lg = create_displaced_cells(
                cell, mu_index, params['grid_n'], 3*em[i]*Ri)
            write_displaced_cells(cell, sname, pname, lg, i)

    else:
        # Parse hyperfine values from .magres files and energy from .castep
        # files
        E_table = []
        hfine_table = np.zeros((np.size(R), params['grid_n']))
        ipso_hfine_table = np.zeros((np.size(R), params['grid_n']))
        num_species = np.size(cell.get_array('castep_custom_species'))
        all_hfine_tensors = np.zeros(
            (num_species, np.size(R), params['grid_n'], 3, 3))
        for i, Ri in enumerate(R):
            E_table.append([])
            dirname = '{0}_{1}'.format(sname, i+1)
            for j in range(params['grid_n']):
                mfile = os.path.join(
                    dirname, '{0}_{1}_{2}.magres'.format(sname, i+1, j+1))
                mgr = parse_hyperfine_magres(mfile)
                hfine_table[i][j] = np.trace(
                    mgr.get_array('hyperfine')[mu_index])/3.0
                if not params['ignore_ipsoH']:
                    ipso_hfine_table[i][j] = np.trace(
                        mgr.get_array('hyperfine')[ipso_H_index])/3.0
                else:
                    ipso_hfine_table = None
                for k, tensor in enumerate(mgr.get_array('hyperfine')):
                    all_hfine_tensors[k][i][j][:][:] = tensor
                castf = os.path.join(
                    dirname, '{0}_{1}_{2}.castep'.format(sname, i+1, j+1))
                E_table[-1].append(parse_final_energy(castf))

        E_table = np.array(E_table)
        if (hfine_table.shape != (3, params['grid_n']) or
                E_table.shape != (3, params['grid_n'])):
            raise RuntimeError("Incomplete or absent magres or castep data")

        symbols = cell.get_array('castep_custom_species')

        r2psi2 = calc_wavefunction(R, params['grid_n'], mu_mass, E_table,
                                   hfine_table, sname,
                                   params['numerical_solver'], True)
        D1, D2, ipso_D1, ipso_D2 = avg_hfine_tensor(r2psi2, hfine_table,
                                                    all_hfine_tensors[
                                                        mu_index],
                                                    params['ignore_ipsoH'],
                                                    ipso_hfine_table,
                                                    all_hfine_tensors[
                                                        ipso_H_index],
                                                    sname)
        if (params['save_tensors']):
            write_tensors(sname, all_hfine_tensors, r2psi2, symbols)
        calc_harm_potential(R, params['grid_n'],
                            mu_mass, evals, E_table, sname)

    return
