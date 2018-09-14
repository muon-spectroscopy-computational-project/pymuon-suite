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
    #Grab muon mass
    lines =  open(params['phonon_file'] + '.phonon').readlines()
    for i in range(len(lines)):
        if "Fractional Co-ordinates" in lines[i]:
            mu_mass = lines[i+1+mu_index].split()[5]
            try:
                mu_mass = float(mu_mass)
            except ValueError:
                print(".phonon file does not contain muon mass")
            break

    #Find ipso hydrogen
    if not 'True' in params['ignore_ipsoH']:
        ipso_H_index = find_ipso_hydrogen(mu_index, cell, params['muon_symbol'])


    return
