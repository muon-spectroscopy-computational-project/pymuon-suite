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
import warnings
from copy import deepcopy

import numpy as np
import argparse as ap
from spglib import find_primitive

from ase import Atoms, io
from ase.io.castep import read_param
from ase.build import make_supercell
from ase.calculators.castep import Castep
from ase.calculators.dftb import Dftb
from soprano.utils import safe_input, customize_warnings
from soprano.collection import AtomsCollection
from soprano.collection.generate import defectGen
from soprano.analyse.phylogen import PhylogenCluster, Gene

import pymuonsuite.constants as cnst
from pymuonsuite.utils import make_3x3, safe_create_folder, list_to_string
from pymuonsuite.schemas import load_input_file, MuAirssSchema
from pymuonsuite.io.castep import add_to_castep_block, ReadWriteCastep
from pymuonsuite.io.dftb import ReadWriteDFTB
from pymuonsuite.io.uep import UEPCalculator, ReadWriteUEP
from pymuonsuite.io.output import write_cluster_report

customize_warnings()

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


def save_muairss_collection(struct, params, batch_path=''):
    """Generate input files for a single structure and configuration file"""

    dc = generate_muairss_collection(struct, params)
    # Just to keep track, add the parameters used to the collection
    dc.info['muairss_params'] = dict(params)

    # Output folder
    out_path = safe_create_folder(os.path.join(batch_path,
                                  params['out_folder']))

    if not out_path:
        raise RuntimeError('Could not create folder {0}')

    io_formats = {
        'castep': ReadWriteCastep(params),
        'dftb+': ReadWriteDFTB(params),
        'uep': ReadWriteUEP(params)
    }

    calcs = [s.strip().lower() for s in params['calculator'].split(',')]
    if 'all' in calcs:
        calcs = io_formats.keys()

    # Save LICENSE file for DFTB+ parameters
    if 'dftb+' in calcs:
        from pymuonsuite.data.dftb_pars import get_license
        with open(os.path.join(out_path, 'dftb.LICENSE'), 'w') as f:
            f.write(get_license())

    for cname in calcs:
        calc_path = os.path.join(out_path, cname)
        dc.save_tree(calc_path, io_formats[cname].write,
                     name_root=params['name'],
                     opt_args={'calc_type': "GEOM_OPT"},
                     safety_check=2)


def load_muairss_collection(struct, params, batch_path=''):

    # Output folder
    out_path = os.path.join(batch_path, params['out_folder'])

    load_formats = {
        'castep': ReadWriteCastep(),
        'dftb+': ReadWriteDFTB(),
        'uep': ReadWriteUEP()
    }

    calcs = [s.strip().lower() for s in params['calculator'].split(',')]
    if 'all' in calcs:
        calcs = load_formats.keys()

    loaded = {}

    for cname in calcs:
        calc_path = os.path.join(out_path, cname)

        dc = AtomsCollection.load_tree(calc_path, load_formats[cname].read,
                                       safety_check=2, tolerant=True)

        print("If greater than 10% of structures could not be loaded, \
we advise adjusting the parameters and re-running the {0} \
optimisation for the structures that failed.".format(cname))

        total_structures = len(dc.structures)
        if total_structures == 0:
            return

        loaded[cname] = dc

    return loaded


def muairss_batch_io(args, global_params, save=False):
    structures_path = args.structures

    all_files = glob.glob(os.path.join(structures_path, "*"))
    structure_files = [path for path in all_files
                       if not os.path.splitext(path)[1] == '.yaml']

    if save:
        global_params['out_folder'] = safe_create_folder(
            global_params['out_folder'])

    print("Beginning {0} of {1} structures".format(
        'creation' if save else 'loading', len(structure_files)))

    bpath = global_params['out_folder']

    loaded = {}

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

        if save:
            print("Making {} ---------------------".format(name))
            save_muairss_collection(struct, params,
                                    batch_path=bpath)
        else:
            print("Loading {} ---------------------".format(name))
            coll = load_muairss_collection(struct, params,
                                           batch_path=bpath)
            loaded[name] = {'struct': struct, 'collection': coll}

    print("Done!")

    if not save:
        return loaded


def muairss_cluster(struct, collection, params, name=None):

    if name is None:
        name = params['name']

    clusters = {}

    for calc, ccoll in collection.items():
        # First, filter out all failed results
        def calc_filter(a):
            return a.calc is not None

        n = len(ccoll)
        ccoll = ccoll.filter(calc_filter)
        if len(ccoll) < n:
            warnings.warn('Calculation failed for {0}% of structures.'
                          ' If greater than 10% of the calculations failed,'
                          ' we advise adjusting the parameters and re-running'
                          ' the optimisation for the runs that failed.'.format(
                           round((1-len(ccoll)/n)*100)))

        # Start by extracting the muon positions
        genes = [Gene('energy', 1, {}),
                 Gene('defect_asymmetric_fpos', 1,
                      {'index': -1, 'struct': struct})]
        pclust = PhylogenCluster(ccoll, genes=genes)

        cmethod = params['clustering_method']
        if cmethod == 'hier':
            cl = pclust.get_hier_clusters(params['clustering_hier_t'])
        elif cmethod == 'kmeans':
            cl = pclust.get_kmeans_clusters(params['clustering_kmeans_k'])

        # Get the energy
        gvecs = pclust.get_genome_vectors()[0]
        # Split the collection
        clusters[calc] = [cl, ccoll.classify(cl[0]), gvecs]

    return clusters


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
            muairss_batch_io(args, params, True)
        elif os.path.isfile(args.structures):
            struct = io.read(args.structures)
            save_muairss_collection(struct, params)
        else:
            raise RuntimeError("{} is neither a file or a directory"
                               .format(args.structures))
    elif task == 'r':
        if os.path.isdir(args.structures):
            all_coll = muairss_batch_io(args, params)
            clusters = {}
            for name, data in all_coll.items():
                clusters[name] = muairss_cluster(data['struct'],
                                                 data['collection'], params)
        elif os.path.isfile(args.structures):
            struct = io.read(args.structures)
            collection = load_muairss_collection(struct, params)
            clusters = {
                params['name']: muairss_cluster(struct, collection, params)
            }
        else:
            raise RuntimeError("{} is neither a file or a directory"
                               .format(args.structures))
        write_cluster_report(args, params, clusters)
