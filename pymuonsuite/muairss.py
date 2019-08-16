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
from copy import deepcopy

import numpy as np
import argparse as ap
from spglib import find_primitive

from ase import Atoms, io
from ase.io.castep import read_param
from ase.build import make_supercell
from ase.calculators.castep import Castep
from ase.calculators.dftb import Dftb
from soprano.utils import safe_input
from soprano.collection import AtomsCollection
from soprano.collection.generate import defectGen

import pymuonsuite.constants as cnst
from pymuonsuite.utils import make_3x3, safe_create_folder, list_to_string
from pymuonsuite.schemas import load_input_file, MuAirssSchema
from pymuonsuite.io.castep import (castep_write_input, add_to_castep_block)
from pymuonsuite.io.dftb import dftb_write_input, dftb_read_input
from pymuonsuite.io.uep import UEPCalculator, uep_write_input


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
    smdiag = np.diag(sm).astype(int)
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
        scell.set_pbc(params['dftb_pbc'])
        collection.append(scell)

    return AtomsCollection(collection)


def safe_create_folder(folder_name):
    while os.path.isdir(folder_name):
        ans = safe_input(('Folder {} exists, overwrite (y/N)? '
                          ).format(folder_name))
        if ans == 'y':
            shutil.rmtree(folder_name)
        else:
            folder_name = safe_input('Please input new folder name:\n')
    try:
        os.mkdir(folder_name)
    except OSError:
        pass  # It's fine, it already exists
    return folder_name


def parse_structure_name(file_name):
    name = os.path.basename(file_name)
    base = os.path.splitext(name)[0]
    return base


def create_muairss_castep_calculator(a, params={}, calc=None):
    """Create a calculator containing all the necessary parameters
    for a geometry optimization."""

    if not isinstance(calc, Castep):
        calc = Castep()
    else:
        calc = deepcopy(calc)

    musym = params.get('mu_symbol', 'H:mu')

    # Start by ensuring that the muon mass and gyromagnetic ratios are included
    mass_block = calc.cell.species_mass.value
    calc.cell.species_mass = add_to_castep_block(mass_block, musym,
                                                 cnst.m_mu_amu,
                                                 'mass')

    gamma_block = calc.cell.species_gamma.value
    calc.cell.species_gamma = add_to_castep_block(gamma_block, musym,
                                                  851586494.1, 'gamma')

    # Now assign the k-points
    calc.cell.kpoint_mp_grid = list_to_string(
        params.get('k_points_grid', [1, 1, 1]))
    calc.cell.fix_all_cell = True   # Necessary for older CASTEP versions
    calc.param.charge = params.get('charged', False)*1.0

    # Read the parameters
    pfile = params.get('castep_param', None)
    if pfile is not None:
        calc.param = read_param(params['castep_param']).param

    calc.param.task = 'GeometryOptimization'
    calc.param.geom_max_iter = params.get('geom_steps', 30)
    calc.param.geom_force_tol = params.get('geom_force_tol', 0.05)
    calc.param.max_scf_cycles = params.get('max_scc_steps', 30)

    return calc


def create_muairss_dftb_calculator(a, params={}, calc=None):

    from pymuonsuite.data.dftb_pars.dftb_pars import DFTBArgs

    if not isinstance(calc, Dftb):
        args = {}
    else:
        args = calc.todict()

    dargs = DFTBArgs(params['dftb_set'])

    for opt in params['dftb_optionals']:
        try:
            dargs.set_optional(opt, True)
        except KeyError:
            print(('WARNING: optional DFTB+ file {0} not available for {1}'
                   ' parameter set, skipping').format(opt, params['dftb_set'])
                  )

    args.update(dargs.args)
    args = dargs.args
    args['Driver_'] = 'ConjugateGradient'
    args['Driver_Masses_'] = ''
    args['Driver_Masses_Mass_'] = ''
    args['Driver_Masses_Mass_Atoms'] = '-1'
    args['Driver_Masses_Mass_MassPerAtom [amu]'] = str(cnst.m_mu_amu)

    args['Driver_MaxForceComponent [eV/AA]'] = params['geom_force_tol']
    args['Driver_MaxSteps'] = params['geom_steps']
    args['Driver_MaxSccIterations'] = params['max_scc_steps']
    args['Hamiltonian_Charge'] = 1.0 if params['charged'] else 0.0

    if params['dftb_pbc']:
        calc = Dftb(kpts=params['k_points_grid'],
                    run_manyDftb_steps=True, **args)
    else:
        calc = Dftb(run_manyDftb_steps=True, **args)

    return calc


def create_muairss_uep_calculator(a, params={}, calc=None):

    if not isinstance(calc, UEPCalculator):
        calc = UEPCalculator(atoms=a, chden=params['uep_chden'])
    else:
        dummy = UEPCalculator(chden=params['uep_chden'])
        calc.chden_path = dummy.chden_path
        calc.chden_seed = dummy.chden_seed

    if not params['charged']:
        raise RuntimeError("Can't use UEP method for neutral system")

    calc.label = params['name']
    calc.gw_factor = params['uep_gw_factor']
    calc.geom_steps = params['geom_steps']
    calc.opt_tol = params['geom_force_tol']

    return calc


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
        'castep': castep_write_input,
        'dftb+': dftb_write_input,
        'uep': uep_write_input
    }

    # Which calculators?
    calcs = [s.strip().lower() for s in params['calculator'].split(',')]
    if 'all' in calcs:
        calcs = save_formats.keys()

    # Make the actual calculators
    make_calcs = {
        'castep': create_muairss_castep_calculator,
        'dftb+': create_muairss_dftb_calculator,
        'uep': create_muairss_uep_calculator
    }

    calcs = {c: make_calcs[c](struct, params=params, calc=struct.calc)
             for c in calcs}

    # Save LICENSE file for DFTB+ parameters
    if 'dftb+' in calcs:
        from pymuonsuite.data.dftb_pars import get_license
        with open(os.path.join(out_path, 'dftb.LICENSE'), 'w') as f:
            f.write(get_license())

    for cname, calc in calcs.items():
        calc_path = os.path.join(out_path, cname)
        dc.save_tree(calc_path, save_formats[cname], name_root=params['name'],
                     opt_args={'calc': calc, 'script': params['script_file']},
                     safety_check=2)


def load_muairss_collection(struct, params, batch_path=''):

    # Output folder
    out_path = os.path.join(batch_path, params['out_folder'])

    load_formats = {
        'dftb+': dftb_read_input
    }

    calcs = [s.strip().lower() for s in params['calculator'].split(',')]
    if 'all' in calcs:
        calcs = load_formats.keys()

    loaded = {}

    for cname in calcs:
        calc_path = os.path.join(out_path, cname)
        dc = AtomsCollection.load_tree(calc_path, load_formats[cname],
                                       safety_check=2)
        loaded[cname] = dc

    return loaded

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


def main_generate():
    main('w')


def main(task=None):
    parser = ap.ArgumentParser()
    parser.add_argument('structures', type=str, default=None,
                        help="A structure file or a folder of files in an ASE "
                        "readable format")
    parser.add_argument('parameter_file', type=str, default=None, help="""YAML
                        formatted file with generation parameters. The
                        arguments can be overridden by structure-specific YAML
                        files if a folder is passed as the first argument.""")
    parser.add_argument('-t', type=str, default='r', choices=['r', 'w'],
                        dest='task',
                        help="""Task to be run by muairss. Can be either 'w'
                        (=generate and WRITE structures) or 'r' (=READ and
                        cluster results). Default is READ.""")

    args = parser.parse_args()
    params = load_input_file(args.parameter_file, MuAirssSchema)

    if task is None:
        task = args.task

    if task == 'w':
        if os.path.isdir(args.structures):
            save_muairss_batch(args, params)
        elif os.path.isfile(args.structures):
            struct = io.read(args.structures)
            save_muairss_collection(struct, params)
        else:
            raise RuntimeError("{} is neither a file or a directory"
                               .format(args.structures))
    elif task == 'r':
        if os.path.isdir(args.structures):
            load_muairss_batch(args, params)
        elif os.path.isfile(args.structures):
            struct = io.read(args.structures)
            load_muairss_collection(struct, params)
        else:
            raise RuntimeError("{} is neither a file or a directory"
                               .format(args.structures))
