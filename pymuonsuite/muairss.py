"""
muairss.py

Utility functions and main script for AIRSS structure generation for muon site
finding.
"""

# Python 2-to-3 compatibility code
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import glob
import shutil

import numpy as np
import argparse as ap
from spglib import find_primitive

from pymuonsuite.io.castep import save_muonconf_castep
from pymuonsuite.io.dftb import save_muonconf_dftb
from pymuonsuite.utils import make_3x3, safe_create_folder
from pymuonsuite.data.dftb_pars.dftb_pars import get_license
from pymuonsuite.schemas import load_input_file, MuAirssSchema

from ase import Atoms, io
from ase.build import make_supercell
from soprano.collection import AtomsCollection
from soprano.collection.generate import defectGen


def find_primitive_structure(struct):
    """Find the structure contained within the reduced cell

    | Args:
    |   struct (ase.Atoms): structure to find the reduced cell for.
    |
    | Returns:
    |   reduced_struct(ase.Atoms): the structure in the reduced cell.

    """
    params = (struct.cell, struct.get_scaled_positions(), struct.numbers)
    lattice, scaled_positions, numbers = find_primitive(params)
    reduced_struct = Atoms(cell=lattice, scaled_positions=scaled_positions,
                           numbers=numbers)
    return reduced_struct


def generate_muairss_collection(struct, params):
    if params['mu_symbol'] in struct.get_chemical_symbols():
        print('WARNING: chosen muon symbol conflicts with existing elements in'
              ' the starting unit cell. This could cause mistakes')

    # Make a supercell
    sm = make_3x3(params['supercell'])
    # ASE's make_supercell is weird, avoid if not necessary...
    smdiag = np.diag(sm)
    if np.all(np.diag(smdiag) == sm):
        scell0 = struct.repeat(smdiag)
    else:
        scell0 = make_supercell(struct, sm)

    reduced_struct = find_primitive_structure(struct)

    print('Generating defect configurations...')
    # Now generate the defect configurations
    defect_gen = defectGen(reduced_struct, 'H', poisson_r=params['poisson_r'],
                           vdw_scale=params['vdw_scale'])
    defect_collection = AtomsCollection(defect_gen)
    print('{0} configurations generated'.format(len(defect_collection)))

    collection = []
    for atoms in defect_collection:
        # Where's the muon?
        # We rely on the fact that it's always put at the first place
        mupos = atoms.get_positions()[0]
        scell = scell0.copy() + Atoms('H', positions=[mupos])
        # Add castep custom species
        csp = scell0.get_chemical_symbols() + [params['mu_symbol']]
        scell.set_array('castep_custom_species', np.array(csp))
        collection.append(scell)

    return AtomsCollection(collection)


def save_muairss_collection(struct, params, batch_path=''):
    """Generate input files for a single structure and configuration file"""

    dc = generate_muairss_collection(struct, params)
    # Just to keep track, add the parameters used to the collection
    dc.info['muairss_params'] = dict(params)

    # Output folder
    out_path = os.path.join(batch_path, params['out_folder'])

    if not safe_create_folder(out_path):
        raise RuntimeError('Could not create folder {0}')

    # Now save in the appropriate format
    save_formats = {
        'castep': save_muonconf_castep,
        'dftb+': save_muonconf_dftb
    }

    # Which calculators?
    calcs = map(lambda s: s.strip(), params['calculator'].split(','))
    if 'all' in calcs:
        calcs = save_formats.keys()

    # Save LICENSE file for DFTB+ parameters
    if 'dftb+' in calcs:
        with open(os.path.join(out_path, 'dftb.LICENSE'), 'w') as f:
            f.write(get_license())

    for cname in calcs:
        calc_path = os.path.join(out_path, cname)
        dc.save_tree(calc_path, save_formats[cname], name_root=params['name'],
                     opt_args={'params': params}, safety_check=2)


def safe_create_folder(folder_name):
    while os.path.isdir(folder_name):
        ans = raw_input(('Folder {} exists, overwrite (y/N)? '
                         ).format(folder_name))
        if ans == 'y':
            shutil.rmtree(folder_name)
        else:
            folder_name = raw_input('Please input new folder name:\n')
    try:
        os.mkdir(folder_name)
    except OSError:
        pass  # It's fine, it already exists
    return folder_name


def parse_structure_name(file_name):
    name = os.path.basename(file_name)
    base = os.path.splitext(name)[0]
    return base


def save_muairss_batch(args, global_params):
    structures_path = args.structures

    all_files = glob.glob(os.path.join(structures_path, "*"))
    structure_files = [path for path in all_files
                       if not os.path.splitext(path)[1] == '.yaml']

    global_params['out_folder'] = safe_create_folder(
        global_params['out_folder'])

    print("Beginning creation of {} structures".format(len(structure_files)))

    for path in structure_files:
        name = parse_structure_name(path)
        parameter_file = os.path.join(structures_path, "{}.yaml".format(name))
        if not os.path.isfile(parameter_file):
            parameter_file = None

        struct = io.read(path)
        params = dict(global_params)    # Copy
        params['name'] = name
        if parameter_file is not None:
            params = load_input_file(parameter_file, MuAirssSchema,
                                     merge=params)
        params['out_folder'] = params['name']

        print("Making {} ---------------------".format(name))
        save_muairss_collection(struct, params,
                                batch_path=global_params['out_folder'])

    print("Done!")


def main():
    parser = ap.ArgumentParser()
    parser.add_argument('structures', type=str, default=None,
                        help="A structure file or a folder of files in an ASE "
                        "readable format")
    parser.add_argument('parameter_file', type=str, default=None, help="""YAML
                        formatted file with generation parameters. The 
                        arguments can be overridden by structure-specific YAML
                        files if a folder is passed as the first argument.""")

    args = parser.parse_args()
    params = load_input_file(args.parameter_file, MuAirssSchema)

    if os.path.isdir(args.structures):
        save_muairss_batch(args, params)
    elif os.path.isfile(args.structures):
        struct = io.read(args.structures)
        save_muairss_collection(struct, params)
    else:
        raise RuntimeError("{} is neither a file or a directory"
                           .format(args.structures))
