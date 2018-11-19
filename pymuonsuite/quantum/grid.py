"""
Author: Simone Sturniolo(Functionality) and Adam Laverack(Interface)
"""

# Python 2-to-3 compatibility code
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import scipy.constants as cnst
from ase import Atoms


def calc_wavefunction(R, grid_n, mu_mass, E_table, hfine_table, sname = None,
                        num_solve = False, write_table = True):
    """
    Calculate wavefunction

    | Args:
    |   R (Numpy float array): Displacement factors along all phonon axes
    |   grid_n (int): Number of displacements per axis
    |   mu_mass (float): Mass of muon
    |   E_table (Numpy float array, shape:(size(R), grid_n)): Table of system
    |   energies for each muon displacement for each axis.
    |   hfine_table (Numpy float array, shape:(size(R), grid_n)): Table of
    |   hyperfine coupling constants
    |   sname (str): Seedname of file to write to
    |   num_solve (bool): Solve schroedinger equation numerically using qlab
    |   write_table: Write out table of wavefunction values

    """
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
