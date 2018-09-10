#!/usr/bin/env python

# Python 2-to-3 compatibility code
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import shutil
import numpy as np
import scipy.constants as cnst
import argparse as ap
from ase import io as ase_io
from ase.io.magres import read_magres
from soprano.utils import seedname, minimum_periodic
from soprano.collection.generate import linspaceGen

def find_muon_ipso_hydrogen(pos, cell_0, noH, symbol):
    """
    Return index of muon in cell postion array. Also return index of nearest
    hydrogen unless noH flag set.
    """
    #Find the muon
    a_i = -1
    if cell_0.has('castep_custom_species'):
        chems = cell_0.get_array('castep_custom_species')
    else:
        chems = np.array(cell_0.get_chemical_symbols())
    a_i = np.where(chems == params['symbol'])[0]

    if not noH:
        #Find the closest hydrogen
        iH = np.where(['H' in c and c != params['symbol'] for c in chems])[0]
        posH = pos[iH]
        distH = np.linalg.norm(
            minimum_periodic(posH - pos[a_i], cell_0.get_cell())[0], axis=-1)
        #Which one is the closest?
        ipso_i = iH[np.argmin(distH)]
    else:
        ipso_i = None

    return a_i, ipso_i

def parse_input_file(infile):

    lines = open(infile).readlines()
    # Remove comments
    for i in range(len(lines)-1, -1, -1):
        lines[i] = lines[i].split('#')[0]
        if len(lines[i]) == 0:
            del(lines[i])

    data = {}
    data['m'] = float(lines[0].strip().split()[0])
    data['omega'] = np.array(
        [float(x) for x in lines[1].strip().split()])*1e2*cnst.c*np.pi*2
    data['evecs'] = np.array(
        [[float(x) for x in l.strip().split()] for l in lines[2:5]])
    data['symbol'] = lines[5].strip()
    data['grid_n'] = int(lines[6].strip())

    return data


def parse_hyperfine_magres(infile):

    # First, `simply parse the magres file via ASE
    mgr = read_magres(infile, True)

    # Now go for the magres_old block

    if 'magresblock_magres_old' not in mgr.info:
        raise RuntimeError('.magres file has no hyperfine information')

    hfine = parse_hyperfine_oldblock(mgr.info['magresblock_magres_old'])

    labels, indices = mgr.get_array('labels'), mgr.get_array('indices')

    hfine_array = []
    for l, i in zip(labels, indices):
        hfine_array.append(hfine[l][i])

    mgr.new_array('hyperfine', np.array(hfine_array))

    return mgr


def parse_hyperfine_oldblock(block):

    hfine_dict = {}

    sp = None
    n = None
    tens = None
    block_lines = block.split('\n')
    for i, l in enumerate(block_lines):
        if 'Atom:' in l:
            # Get the species and index
            _, sp, n = l.split()
            n = int(n)
        if 'TOTAL tensor' in l:
            tens = np.array([[float(x) for x in row.split()]
                             for row in block_lines[i+2:i+5]])
            # And append
            if sp is None:
                raise RuntimeError('Invalid block in magres hyperfine file')
            if sp not in hfine_dict:
                hfine_dict[sp] = {}
            hfine_dict[sp][n] = tens

    return hfine_dict

def write_displaced_cells(pos, cell_0, R, sname, pname, a_i, evecs, grid_n):
    """
    Write grid_n cell files for each eigenvector with a range of muon
    displacements from -3*R*evec to +3*R*evec
    """
    for i, Ri in enumerate(R):
        dirname = '{0}_{1}'.format(sname, i+1)
        print("Creating folder", dirname)
        try:
            os.mkdir(dirname)
        except OSError:
            # Folder already exists
            pass
        cell_L = cell_0.copy()
        pos_L = pos.copy()
        pos_L[a_i] -= evecs[i]*3*Ri
        cell_L.set_positions(pos_L)
        cell_R = cell_0.copy()
        pos_R = pos.copy()
        pos_R[a_i] += evecs[i]*3*Ri
        cell_R.set_positions(pos_R)
        lg = linspaceGen(
            cell_L, cell_R, steps=grid_n, periodic=True)
        for j, c in enumerate(lg):
            c.set_calculator(cell_0.calc)
            ase_io.write(os.path.join(dirname,
                                      '{0}_{1}_{2}.cell'.format(sname, i+1, j+1)), c)
            # If present, copy param file!
            try:
                shutil.copy(pname, os.path.join(dirname,
                                                '{0}_{1}_{2}.param'.format(sname, i+1, j+1)))
            except IOError:
                pass
    return

if __name__ == "__main__":

    parser = ap.ArgumentParser()
    # Necessary arguments
    parser.add_argument(
        'cell_file', type=str, default=None, help="File containing the base CELL")
    parser.add_argument('modes_file', type=str, default=None,
                        help="File containing input information about phonon modes")
    # Optional arguments
    parser.add_argument('-w',   action='store_true', default=False,
                        help="Create and write input files instead of parsing the results")
    parser.add_argument('-tens',   action='store_true', default=False,
                        help="Save also all the tensors")
    parser.add_argument('-num',   action='store_true', default=False,
                        help="Solve the Schroedinger equation numerically on the three axes")
    parser.add_argument('-noH',   action='store_true', default=False,
                        help="Ignore the closest hydrogen, only do the labelled atom")

    args = parser.parse_args()

    cell_0 = ase_io.read(args.cell_file)
    params = parse_input_file(args.modes_file)

    # Displacement in Angstrom
    R = np.sqrt(cnst.hbar/(params['omega']*params['m']))*1e10

    pos = cell_0.get_positions()
    #Find muon and closest hydrogen, if noH flag not set
    a_i, ipso_i = find_muon_ipso_hydrogen(pos, cell_0, args.noH, params['symbol'])

    if len(a_i) != 1:
        raise RuntimeError(
            'Invalid element symbol in {0}'.format(args.modes_file))
    else:
        a_i = a_i[0]

    sname = seedname(args.cell_file)
    pname = os.path.splitext(args.cell_file)[0] + '.param'
    if not os.path.isfile(pname):
        print("WARNING - no .param file was found")

    if args.w:
        #Write cells with a range of muon displacements
        write_displaced_cells(pos, cell_0, R, sname, pname, a_i, params['evecs'], params['grid_n'])

    else:
        # Parsing

        all_hfine_tensors = {}
        E_table = []
        for i, Ri in enumerate(R):
            E_table.append([])
            dirname = '{0}_{1}'.format(sname, i+1)
            for j in range(params['grid_n']):
                mfile = os.path.join(
                    dirname, '{0}_{1}_{2}.magres'.format(sname, i+1, j+1))
                # Parse the hyperfine coupling value
                mlines = open(mfile).readlines()
                for l_i, l in enumerate(mlines):
                    # Also grab the eigenvalues
                    try:
                        sym, ind = l.split()[:2]
                        ind = int(ind)
                    except ValueError:
                        continue

                    if 'Coordinates' in l:
                        if (sym, ind) not in all_hfine_tensors:
                            all_hfine_tensors[(sym, ind)] = np.zeros((len(R), params['grid_n'], 3, 3))
                        hfine_tens = [[float(x) for x in row.split()]
                                       for row in mlines[l_i+4:l_i+7]]
                        all_hfine_tensors[(sym, ind)][i,j] = hfine_tens

                # Also grab the energy
                castf = os.path.join(
                    dirname, '{0}_{1}_{2}.castep'.format(sname, i+1, j+1))
                E = None
                for l in open(castf).readlines():
                    if "Final energy" in l:
                        try:
                            E = float(l.split()[3])
                        except ValueError:
                            raise RuntimeError(
                                "Corrupt .castep file found: {0}".format(castf))
                E_table[-1].append(E)

        hfine_tensors = all_hfine_tensors[(params['symbol'], 1)]
        hfine_table = np.trace(hfine_tensors, axis1=-1, axis2=-2)/3.0

        ipso_hfine_tensors = all_hfine_tensors[('H', ipso_i+1)]
        ipso_hfine_table = np.trace(ipso_hfine_tensors, axis1=-1,
                                    axis2=-2)/3.0

        E_table = np.array(E_table)
        if hfine_table.shape != (3, params['grid_n']) or E_table.shape != (3, params['grid_n']):
            raise RuntimeError("Incomplete or absent magres or castep data")

        R_axes = np.array([np.linspace(-3*Ri, 3*Ri, params['grid_n'])
                           for Ri in R])

        if not args.num:
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
            sname += '_num'
            psi = []
            for i, Ri in enumerate(R):
                qSol = QSolution([(-3e-10*Ri, 3e-10*Ri)], params['grid_n'],
                                 E_table[i]*cnst.electron_volt, params['m'])
                psi.append(qSol.evec_grid(0))
            psi = np.array(psi)
            # Oh, and save the densities!

        psi_table = np.concatenate(
            (R_axes, E_table, psi**2, hfine_table), axis=0)
        np.savetxt(sname + '_psi.dat', psi_table.T)
        # Output
        ofile = open(sname + '_report.txt', 'w')
        # And average
        r2psi2 = R_axes**2*np.abs(psi)**2

        hfine_avg = np.sum(r2psi2*hfine_table)/np.sum(r2psi2)
        ofile.write('Predicted hyperfine coupling on labeled atom ({1}): {0} MHz\n'.format(
            hfine_avg, params['symbol']))

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

        ofile.write(('Predicted dipolar hyperfine components on labeled atom ({2}):\n'
                     'D1:\t{0} MHz\nD2:\t{1} MHz\n').format(
            D1, D2, params['symbol']))

        if not args.noH:
            ipso_avg = np.sum(r2psi2*ipso_hfine_table)/np.sum(r2psi2)
            ofile.write('Predicted hyperfine coupling on closest hydrogen (H_{1}): {0} MHz\n'.format(
                ipso_avg, ipso_i+1))

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
                D1 = evals_notr[2]
                D2 = evals_notr[1]-evals_notr[0]
            else:
                D1 = evals_notr[0]
                D2 = evals_notr[2]-evals_notr[1]

            ofile.write(('Predicted dipolar hyperfine components on closest hydrogen (H_{2}):\n'
                         'D1:\t{0} MHz\nD2:\t{1} MHz\n').format(
                D1, D2, ipso_i+1))

        if args.tens:
            # Also save tensor file
            tensfile = open(sname + '_tensors.dat', 'w')
            for sym, ind in all_hfine_tensors:
                hfine_tensors_i = all_hfine_tensors[(sym, ind)]
                # Carry out the average
                hfine_avg = np.sum(
                r2psi2[:, :, None, None]*hfine_tensors_i, axis=(0, 1))/np.sum(r2psi2)
                tensfile.write('{0}_{1}\n'.format(sym, ind))
                tensfile.write('\n'.join(['\t'.join([str(x) for x in l]) for l in hfine_avg]) + '\n')

        # Now the potential, measured vs. theoretical
        harm_K = params['m']*params['omega']**2
        harm_V = (0.5*harm_K[:, None]*(R_axes*1e-10)**2)/cnst.electron_volt
        # Normalise E_table
        if E_table.shape[1] % 2 == 1:
            E_table -= (E_table[:, E_table.shape[1]/2])[:, None]
        else:
            E_table -= (E_table[:, E_table.shape[1]/2] +
                        E_table[:, E_table.shape[1]/2-1])[:, None]/2.0
        all_table = np.concatenate((R_axes, harm_V, E_table), axis=0)
        np.savetxt(sname + '_V.dat', all_table.T)
