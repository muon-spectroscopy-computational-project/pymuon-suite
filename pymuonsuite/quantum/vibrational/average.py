"""
average.py

Quantum vibrational averages
"""

# Python 2-to-3 compatibility code
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
from ase import io
from soprano.utils import seedname
try:
    from casteppy.data.phonon import PhononData
except ImportError:
    raise ImportError("""
Can't use castep phonon interface due to casteppy not being installed.
Please download and install casteppy from Bitbucket:

HTTPS:  https://bitbucket.org/casteppy/casteppy.git
SSH:    git@bitbucket.org:casteppy/casteppy.git

and try again.""")

# Internal imports
from pymuonsuite.io.castep import parse_castep_masses


def read_castep_phonons(phonon_file):
    # Parse CASTEP phonon data into casteppy object
    pd = PhononData(phonon_file)
    # Convert frequencies back to cm-1
    pd.convert_e_units('1/cm')
    # Get phonon frequencies+modes
    evals = np.array(pd.freqs)
    evecs = np.array(pd.eigenvecs)

    return evals, evecs


def compute_dftbp_phonons(atoms, param_set, kpts):
    pass


def muon_vibrational_average_write(cell_file, method='independent', muon_index=-1,
                                   muon_symbol=None, grid_n=20, property='hyperfine',
                                   param_file=None, phonon_source='castep',
                                   **kwargs):
    """
    Write input files to compute a vibrational average for a quantity on a muon 
    in a given system.

    | Pars:
    |   cell_file (str):    Filename for input structure file
    |   method (str):       Method to use for the average. Options are 'independent',
    |                       'thermal'. Default is 'independent'.
    |   muon_index (int):   Position of the muon in the given cell file. Default is -1.
    |   muon_symbol (str):  If present, use this symbol to look for the muon among
    |                       CASTEP custom species. Overrides muon_index.
    |   grid_n (int):       Number of configurations used for sampling. Applies slightly
    |                       differently to different schemes.
    |   property (str):     Property to calculate and average. Default is 'hyperfine'.
    |   param_file (str):   If present, copy this CASTEP .param file to all folders.
    |   phonon_source (str):Source of the phonon data. Can be 'castep' or 'asedftbp'.
    |                       Default is 'castep'.
    |   **kwargs:           Other arguments (such as specific arguments for the given 
    |                       phonon method)
    """

    # Open the structure file
    cell = io.read(cell_file)
    sname = seedname(cell_file)
    num_atoms = len(cell)

    # Fetch species
    try:
        species = cell.get_array('castep_custom_species')
    except KeyError:
        species = np.array(cell.get_chemical_symbols())
    
    print(cell.calc.cell)
    print(parse_castep_masses(cell))

    print(sname)
    print(species)
    print(cell.get_masses())
