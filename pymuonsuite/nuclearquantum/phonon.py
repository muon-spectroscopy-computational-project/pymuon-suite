# Python 2-to-3 compatibility code
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
from ase import Atoms
from ase import io as ase_io
from casteppy.data.phonon import PhononData
from soprano.selection import AtomSelection

from pymuonsuite.io.castep import parse_phonon_file
from pymuonsuite.schemas import load_input_file, PhononHfccSchema
from pymuonsuite.utils import find_ipso_hydrogen

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
                print(".phonon file does not contain muon mass")
            break

    #Get muon phonon modes
    em_i, em, em_o = get_major_emodes(evecs[0], mu_index)
    em = np.real(em)

    #Find ipso hydrogen location and phonon modes
    if not 'True' in params['ignore_ipsoH']:
        ipso_H_index = find_ipso_hydrogen(mu_index, cell, params['muon_symbol'])
        em_i_H, em_H, em_o_H = get_major_emodes(evecs[0], ipso_H_index)
        em_H = np.real(em_H)


    return
