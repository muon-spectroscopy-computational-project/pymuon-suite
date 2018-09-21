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
from casteppy.data.phonon import PhononData
from soprano.collection.generate import linspaceGen
from soprano.selection import AtomSelection
from soprano.utils import seedname

from pymuonsuite.io.castep import parse_final_energy
from pymuonsuite.io.castep import parse_phonon_file
from pymuonsuite.io.magres import parse_hyperfine_magres
from pymuonsuite.schemas import load_input_file, PhononHfccSchema
from pymuonsuite.utils import find_ipso_hydrogen

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


def get_major_emodes(evecs, i):
    """Find the phonon modes of the muon at index i

    | Args:
    |   evecs (Numpy array): Eigenvectors of phonon modes of molecule in shape
    |                        (num_modes, num_ions, 3)
    |   i (int): Index of muon in position array
    |
    | Returns:
    |   major_evecs_i (int[3]): Indices of eigenvectors in evec array
    |   major_evecs (float[3]): Eigenvectors of muon phonon modes
    |   major_evecs_ortho (float[3]):
    """
    # First, find the eigenmodes whose amplitude is greater for ion i
    evecs_amp = np.linalg.norm(evecs, axis=-1)
    ipr = evecs_amp**4/np.sum(evecs_amp**2, axis=-1)[:,None]**2
    evecs_order = np.argsort(ipr[:,i])

    # How many?
    major_evecs_i = evecs_order[-3:]
    major_evecs = evecs[major_evecs_i,i]
    major_evecs_ortho = np.linalg.qr(major_evecs.T)[0].T

    return major_evecs_i, major_evecs, major_evecs_ortho

def calc_wavefunction(R, grid_n, mu_mass, E_table, hfine_table, sname = None,
                        num_solve = False, write_table = True):
    R_axes = np.array([np.linspace(-3*Ri, 3*Ri, grid_n)
                       for Ri in R])

    if not num_solve:
        # Wavefunction
        psi_norm = (1.0/(np.prod(R)**2*np.pi**3))**0.25
        # And along the three axes
        psi = psi_norm*np.exp(-(R_axes/R[:, None])**2/2.0)
    else:
        # Import qlab
        try:
            from qlab.solve import QSolution
        except ImportError:
            raise RuntimeError('QLab not present on this system, '
                               '-num option is invalid')
        if write_table:
            sname += '_num'
        psi = []
        for i, Ri in enumerate(R):
            qSol = QSolution([(-3e-10*Ri, 3e-10*Ri)], grid_n,
                             E_table[i]*cnst.electron_volt, mu_mass)
            psi.append(qSol.evec_grid(0))
        psi = np.array(psi)
    # Oh, and save the densities!
    if write_table:
        psi_table = np.concatenate(
            (R_axes, E_table, psi**2, hfine_table), axis=0)
        np.savetxt(sname + '_psi.dat', psi_table.T)
    # And average
    r2psi2 = R_axes**2*np.abs(psi)**2

    return r2psi2

def avg_hfine_tensor(r2psi2, hfine_table, hfine_tensors, noH, ipso_hfine_table = None,
                        ipso_hfine_tensors = None, sname = None):
    hfine_avg = np.sum(r2psi2*hfine_table)/np.sum(r2psi2)
    # Now average of dipolar components
    hfine_tens_avg = np.sum(
        r2psi2[:, :, None, None]*hfine_tensors, axis=(0, 1))/np.sum(r2psi2)
    # Diagonalise
    evals, evecs = np.linalg.eigh(hfine_tens_avg)
    evals, evecs = zip(*sorted(zip(evals, evecs), key=lambda x: abs(x[0])))
    evals_notr = -np.array(evals)+np.average(evals)

    if abs(evals_notr[2]) > abs(evals_notr[0]):
        D1 = evals_notr[2]
        D2 = evals_notr[1]-evals_notr[0]
    else:
        D1 = evals_notr[0]
        D2 = evals_notr[2]-evals_notr[1]

    if not noH:
        ipso_avg = np.sum(r2psi2*ipso_hfine_table)/np.sum(r2psi2)

        # Now average of dipolar components
        ipso_hfine_tens_avg = np.sum(
            r2psi2[:, :, None, None]*ipso_hfine_tensors, axis=(0, 1))/np.sum(r2psi2)
        # Diagonalise
        evals, evecs = np.linalg.eigh(ipso_hfine_tens_avg)
        evals, evecs = zip(
            *sorted(zip(evals, evecs), key=lambda x: abs(x[0])))
        evals_notr = -np.array(evals)+np.average(evals)

        # Save the two of them
        np.savetxt(sname + '_tensors.dat', np.concatenate([hfine_tens_avg,
                                                           ipso_hfine_tens_avg]))

        if abs(evals_notr[2]) > abs(evals_notr[0]):
            ipso_D1 = evals_notr[2]
            ipso_D2 = evals_notr[1]-evals_notr[0]
        else:
            ipso_D1 = evals_notr[0]
            ipso_D2 = evals_notr[2]-evals_notr[1]
    else:
        ipso_D1 = None
        ipso_D2 = None
        np.savetxt(sname + '_tensors.dat', hfine_tens_avg)
        
    return D1, D2, ipso_D1, ipso_D2

def write_tensors(sname, all_hfine_tensors, r2psi2, symbols):
    # Also save tensor file
    tensfile = open(sname + '_tensors.dat', 'w')
    for i in range(np.size(all_hfine_tensors, 0)):
        hfine_tensors_i = all_hfine_tensors[i]
        # Carry out the average
        hfine_avg = np.sum(
        r2psi2[:, :, None, None]*hfine_tensors_i, axis=(0, 1))/np.sum(r2psi2)
        tensfile.write('{0} {1}\n'.format(symbols[i], i))
        tensfile.write('\n'.join(['\t'.join([str(x) for x in l]) for l in hfine_avg]) + '\n')

def calc_harm_potential(R, grid_n, mu_mass, freqs, E_table, sname):
    R_axes = np.array([np.linspace(-3*Ri, 3*Ri, grid_n)
                       for Ri in R])
    # Now the potential, measured vs. theoretical
    harm_K = mu_mass*freqs**2
    harm_V = (0.5*harm_K[:, None]*(R_axes*1e-10)**2)/cnst.electron_volt
    # Normalise E_table
    if E_table.shape[1] % 2 == 1:
        E_table -= (E_table[:, E_table.shape[1]//2])[:, None]
    else:
        E_table -= (E_table[:, E_table.shape[1]//2] +
                    E_table[:, E_table.shape[1]//2-1])[:, None]/2.0
    all_table = np.concatenate((R_axes, harm_V, E_table), axis=0)
    np.savetxt(sname + '_V.dat', all_table.T)

def phonon_hfcc(param_file):
    #Load parameters
    params = load_input_file(param_file, PhononHfccSchema)
    #Strip .phonon extension for casteppy compatiblity
    if '.phonon' in params['phonon_file']:
        params['phonon_file'] = (params['phonon_file'])[:-7]
    else:
        raise IOError("Invalid phonon file extension, please use .phonon")
    #Parse phonon data into object
    pd = PhononData(params['phonon_file'])
    #Convert frequencies back to cm-1
    pd.convert_e_units('1/cm')
    #Create eigenvector array that is formatted to work with get_major_emodes.
    evecs = np.zeros((pd.n_qpts, pd.n_branches, pd.n_ions, 3), dtype='complex128')
    for i in range(pd.n_qpts):
        for j in range(pd.n_branches):
            for k in range(pd.n_ions):
                evecs[i][j][k][:] = pd.eigenvecs[i][j*pd.n_ions+k][:]

    #Read in cell file
    cell = ase_io.read(params['cell_file'])
    #Find muon index in structure array
    sel = AtomSelection.from_array(cell, 'castep_custom_species', params['muon_symbol'])
    mu_index = sel.indices[0]
    #Get muon mass
    lines =  open(params['phonon_file'] + '.phonon').readlines()
    for i in range(len(lines)):
        if "Fractional Co-ordinates" in lines[i]:
            mu_mass = lines[i+1+mu_index].split()[5]
            try:
                mu_mass = float(mu_mass)
            except ValueError:
                print("ERROR: .phonon file does not contain valid muon mass")
                raise
            break
    mu_mass = mu_mass*cnst.u #Convert to kg

    #Get muon phonon modes
    em_i, em, em_o = get_major_emodes(evecs[0], mu_index)
    em = np.real(em)

    #Find ipso hydrogen location and phonon modes
    if not params['ignore_ipsoH']:
        ipso_H_index = find_ipso_hydrogen(mu_index, cell, params['muon_symbol'])
        em_i_H, em_H, em_o_H = get_major_emodes(evecs[0], ipso_H_index)
        em_H = np.real(em_H)

    #Get muon phonon frequencies and convert to radians/second
    evals = np.array(pd.freqs[0][em_i]*1e2*cnst.c*np.pi*2)
    # Displacement in Angstrom
    R = np.sqrt(cnst.hbar/(evals*mu_mass))*1e10

    #Get seedname
    sname = seedname(params['cell_file'])
    #Write cells with displaced muon
    if params['write_cells']:
        pname = os.path.splitext(params['cell_file'])[0] + '.param'
        if not os.path.isfile(pname):
            print("WARNING - no .param file was found")
        for i, Ri in enumerate(R):
            lg = create_displaced_cells(cell, mu_index, params['grid_n'], 3*em[i]*Ri)
            write_displaced_cells(cell, sname, pname, lg, i)

    else:
        #Parse hyperfine values from .magres files and energy from .castep files
        E_table = []
        hfine_table = np.zeros((np.size(R), params['grid_n']))
        ipso_hfine_table = np.zeros((np.size(R), params['grid_n']))
        num_species = np.size(cell.get_array('castep_custom_species'))
        all_hfine_tensors = np.zeros((num_species, np.size(R), params['grid_n'], 3, 3))
        for i, Ri in enumerate(R):
            E_table.append([])
            dirname = '{0}_{1}'.format(sname, i+1)
            for j in range(params['grid_n']):
                mfile = os.path.join(
                    dirname, '{0}_{1}_{2}.magres'.format(sname, i+1, j+1))
                mgr = parse_hyperfine_magres(mfile)
                hfine_table[i][j] = np.trace(mgr.get_array('hyperfine')[mu_index])/3.0
                if not params['ignore_ipsoH']:
                    ipso_hfine_table[i][j] = np.trace(mgr.get_array('hyperfine')[ipso_H_index])/3.0
                else:
                    ipso_hfine_table = None
                for k, tensor in enumerate(mgr.get_array('hyperfine')):
                    all_hfine_tensors[k][i][j][:][:] = tensor
                castf = os.path.join(
                    dirname, '{0}_{1}_{2}.castep'.format(sname, i+1, j+1))
                E_table[-1].append(parse_final_energy(castf))

        E_table = np.array(E_table)
        if hfine_table.shape != (3, params['grid_n']) or E_table.shape != (3, params['grid_n']):
            raise RuntimeError("Incomplete or absent magres or castep data")

        symbols = cell.get_array('castep_custom_species')

        r2psi2 = calc_wavefunction(R, params['grid_n'], mu_mass, E_table, hfine_table, sname,
                                False, True)
        D1, D2, ipso_D1, ipso_D2 = avg_hfine_tensor(r2psi2, hfine_table, all_hfine_tensors[mu_index],
                                                params['ignore_ipsoH'], ipso_hfine_table, all_hfine_tensors[ipso_H_index], sname)
        write_tensors(sname, all_hfine_tensors, r2psi2, symbols)
        calc_harm_potential(R, params['grid_n'], mu_mass, evals, E_table, sname)

    return


def write_displaced_cells(cell, sname, pname, lg, i):
    """
    Write out all modified cells in lg using seedname "sname". Also copy
    param file at "pname" if one provided.

    | Args:
    |   cell (ASE Atoms object): Seed cell file, used to set appropriate
    |                           calculator
    |   sname (str): Seedname of cell file e.g. seedname.cell
    |   pname (str): Path of param file to be copies
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
