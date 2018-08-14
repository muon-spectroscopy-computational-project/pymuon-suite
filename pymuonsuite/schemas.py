"""
schemas.py

Python schemas for YAML input files for various scripts
"""

# Python 2-to-3 compatibility code
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import yaml
import numpy as np
from schema import Optional, Schema, SchemaError


def validate_matrix_shape(x):
    x = np.array(x)
    return (x.shape == (1,) or x.shape == (3,) or
            x.shape == (9,) or x.shape == (3, 3))


def validate_supercell(value):
    return (value is None or isinstance(value, int) or
            (validate_matrix_shape(value) and np.array(value).dtype == int))


def validate_all_of(values, options):
    return all([x.strip() in options for x in values.split(',')])

def validate_str(s):
    # Validates a string in a Python 2 and 3 compatible way
    try:
        return isinstance(s, basestring)
    except NameError:
        # It must be Python 3!
        return isinstance(s, str)

def load_input_file(fname, param_schema, merge=None):
    """Load a given input YAML file and validate it with a schema.
    """

    if merge is None:
        params = yaml.load(open(fname, 'r'))
    else:
        new_params = yaml.load(open(fname, 'r'))
        params = dict(merge)
        params.update(new_params)

    try:
        params = param_schema.validate(params)
    except SchemaError as e:
        message = 'Bad formatting in input file {0}\n{1}'.format(fname, e)
        raise RuntimeError(message)

    return params

# Parameter file schemas and defaults
MuAirssSchema = Schema({
    # Name to call the folder for containing each structure. This name will
    # be postfixed with a unique number. e.g. struct_001
    Optional('name', default='struct'):
    validate_str,
    # Calculator to generate structure files for. Must be a comma seperated
    # list of values. Currently supported calculators are CASTEP and DFTB+. Can
    # also pass all as an option to generate files for all calculators.
    Optional('calculator', default='dftb+'):
    lambda values: validate_all_of(values, options=('castep', 'dftb+', 'all')),
    # Command to use to run CASTEP.
    Optional('castep_command', default='castep.serial'):
    validate_str,
    # Command to use to run DFTB+.
    Optional('dftb_command', default='dftb+'):
    validate_str,
    # File path to the CASTEP parameter file.
    Optional('castep_param', default=''):
    validate_str,
    # The parameter set to use for DFTB+.
    Optional('dftb_set', default='3ob-3-1'):
    validate_str,
    # Whether to turn on periodic boundary conditions in DFTB+
    Optional('dftb_pbc', default=True):
    bool,
    # Radius to use when generating muon sites with the possion disk algorithm.
    Optional('poisson_r', default=0.8):
    float,
    # Van der Waals radius to use when generating muon sites. Default is 0.5.
    Optional('vdw_scale', default=0.5):
    float,
    # Supercell size and shape to use. This can either be a single int, a list
    # of three integers or a 3x3 matrix of integers. For a single number a
    # diagonal matrix will be generated with the integer repeated on the
    # diagonals. For a list of three numbers a diagonal matrix will be
    # generated where the digonal elements are set to the list. A matrix will
    # be used direclty as is. Default is a 3x3 indentity matrix.
    Optional('supercell', default=1):
    validate_supercell,
    # List of three integer k-points. Default is [1,1,1].
    Optional('k_points_grid', default=np.ones(3).astype(int)):
    lambda x: np.array(x).shape == (3,) and np.array(x).dtype == int,
    # Name to call the output folder used to store the input files
    # that muon-airss generates.
    Optional('out_folder', default='./muon-airss-out'):
    validate_str,
    # The symbol to use for the muon when writing out the castep custom
    # species.
    Optional('mu_symbol', default='H:mu'):
    validate_str,
    # Maximum number of geometry optimisation steps
    Optional('geom_steps', default=30):
    int,
    # Tolerance on geometry optimisation in units of eV/AA.
    Optional('geom_force_tol', default=0.5):
    float,
    # Max number of SCC steps to perform before giving up. Default is
    # 200 which is also the default for DFTB+.
    Optional('max_scc_steps', default=200):
    int,
})
