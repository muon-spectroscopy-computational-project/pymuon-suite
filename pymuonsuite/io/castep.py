# Python 2-to-3 compatibility code
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import re
import os
import yaml
import glob
import numpy as np
import scipy.constants as cnst
from ase import Atoms
from ase import io
from ase.io.castep import write_param
from ase.calculators.castep import Castep
from soprano.selection import AtomSelection
from soprano.utils import seedname

from pymuonsuite import constants
from pymuonsuite.utils import list_to_string
from pymuonsuite.utils import find_ipso_hydrogen
from pymuonsuite.io.magres import parse_hyperfine_magres


class ReadWriteCastep(object):

    def read(self, folder, sname=None, calc_type="GEOM_OPT",
             avg_prop="hyperfine"):
        """Reads Castep output files.

        | Args:
        |   folder (str):           Path to folder from which to read files.
        |   sname (str):            Seedname to save the files with. If not
        |                           given, use the name of the folder.
        |   calc_type (str):        Castep task performed:
        |                           "GEOM_OPT" or "MAGRES"
        |   avg_prop (str):         Property being averaged in case of reading
        |                           "MAGRES"
        """

        if calc_type == "GEOM_OPT":
            try:
                self.read_castep(folder, sname)
                return self.atoms
            except AttributeError as e:
                print("No castep files were found in {}.".format(folder))

        elif calc_type == "MAGRES":
            try:
                self.read_castep(folder, sname)
                if avg_prop == "hyperfine":
                    self.read_castep_hyperfine_magres(folder, sname)
                return self.atoms
            except AttributeError as e:
                print("No castep files were found in {}.".format(folder))

        elif calc_type == "PHONONS":
            return self.read_castep_gamma_phonons(folder, sname)

    def read_castep(self, folder, sname=None):
        try:
            if sname is not None:
                cfile = os.path.join(folder, sname + '.castep')
            else:
                cfile = glob.glob(os.path.join(folder, '*.castep'))[0]
                sname = seedname(cfile)
            self.atoms = io.read(cfile)
            self.atoms.info['name'] = sname
            return self.atoms
        except (IndexError, OSError):
            print("No .castep files found in {}.".format(folder))

    def read_castep_hyperfine_magres(self, folder, sname=None):
        try:
            if sname is not None:
                mfile = os.path.join(folder, sname + '.magres')
            else:
                mfile = glob.glob(os.path.join(folder, '*.magres'))[0]
            m = parse_hyperfine_magres(mfile)
            self.atoms.arrays.update(m.arrays)
        except (IndexError, OSError):
            print("No .magres files found in {}.".format(folder))
        except AttributeError:  # occurs when self.atoms doesn't exist
            print("Cannot read .magres file without\
 reading .castep file first.")

    def read_castep_gamma_phonons(self, folder, sname=None):
        """Parse CASTEP phonon data into a casteppy object,
        and return eigenvalues and eigenvectors at the gamma point.
        """
        try:
            from euphonic import QpointPhononModes
        except ImportError:
            raise ImportError("""
        Can't use castep phonon interface due to Euphonic not being installed.
        Please download and install Euphonic from Github:

        HTTPS:  https://github.com/pace-neutrons/Euphonic.git
        SSH:    git@github.com:pace-neutrons/Euphonic.git

        and try again.""")

        # Parse CASTEP phonon data into casteppy object
        #
        try:
            if sname is not None:
                pd = QpointPhononModes.from_castep(os.path.join(folder,
                                                   sname + '.phonon'))
            else:
                pd = QpointPhononModes.from_castep(glob.glob(
                                    os.path.join(folder, '*.phonon'))[0])
            # Convert frequencies back to cm-1
            pd.frequencies_unit = '1/cm'
            # Get phonon frequencies+modes
            evals = np.array(pd.frequencies.magnitude)
            evecs = np.array(pd.eigenvectors)

            # Only grab the gamma point!
            gamma_i = None
            for i, q in enumerate(pd.qpts):
                if np.isclose(q, [0, 0, 0]).all():
                    gamma_i = i
                    break

            if gamma_i is None:
                raise MuonAverageError('Could not find gamma point phonons in'
                                       ' CASTEP phonon file')

            return evals[gamma_i], evecs[gamma_i]
        except (IndexError, OSError):
            print("No .phonon files found in {}.".format(folder))

    def write(self, a, folder, sname=None, script=None,
              params={}, calc=None, calc_type="GEOM_OPT"):

        """Writes input files for an Atoms object with a Castep
        calculator.

        | Args:
        |   a (ase.Atoms):          Atoms object to write. Can have a Castep
        |                           calculator attached to carry cell/param
        |                           keywords.
        |   folder (str):           Path to save the input files to.
        |   sname (str):            Seedname to save the files with. If not
        |                           given, use the name of the folder.
        |   script (str):           Path to a file containing a submission
        |                           script to copy to the input folder. The
        |                           script can contain the argument
        |                           {seedname} in curly braces, and it will
        |                           be appropriately replaced.
        |   params (dict)           Contains muon symbol, parameter file,
        |                           k_points_grid.
        |   calc (ase.Calculator):  Calculator to attach to Atoms. If
        |                           present, the pre-existent one will
        |                           be ignored.
        |   calc_type (str):        Castep task which will be performed:
        |                           "GEOM_OPT" or "MAGRES"
        """

        if sname is None:
            sname = os.path.split(folder)[-1]  # Same as folder name

        if calc is not None and isinstance(calc, Castep):
            calc = deepcopy(calc)
        else:
            calc = None

        calc = self.create_calculator(params, calc, calc_type)
        a.set_calculator(calc)

        io.write(os.path.join(folder, sname + '.cell'),
                 a, magnetic_moments='initial')
        write_param(os.path.join(folder, sname + '.param'),
                    a.calc.param, force_write=True)

        if script is not None:
            stxt = open(script).read()
            stxt = stxt.format(seedname=name)
            with open(os.path.join(folder, 'script.sh'), 'w') as sf:
                sf.write(stxt)

            #  make this before making the calc classes

    def create_calculator(self, params={}, calc=None, calc_type=None):
        if calc is None:
            calc = Castep()

        mu_symbol = params.get('mu_symbol', 'H:mu')

        # Start by ensuring that the muon mass and gyromagnetic ratios are
        # included
        gamma_block = calc.cell.species_gamma.value
        calc.cell.species_gamma = add_to_castep_block(gamma_block, mu_symbol,
                                                      constants.m_gamma,
                                                      'gamma')
        mass_block = calc.cell.species_mass.value
        calc.cell.species_mass = add_to_castep_block(mass_block, mu_symbol,
                                                     constants.m_mu_amu,
                                                     'mass')

        # Now assign the k-points
        calc.cell.kpoint_mp_grid = list_to_string(
            params.get('k_points_grid', [1, 1, 1]))

        # Read the parameters
        pfile = params.get('castep_param', None)
        if pfile is not None:
            calc.param = read_param(params['castep_param']).param

        if calc_type == "MAGRES":
            self.create_hfine_castep_calculator(calc)
        elif calc_type == "GEOM_OPT":
            self.create_geom_opt_castep_calculator(calc, params)

        self.calc = calc

        return calc

    def create_hfine_castep_calculator(self, calc):
        """Update calculator to contain all the necessary parameters
        for a hyperfine calculation."""

        calc.param.task = 'Magres'
        calc.param.magres_task = 'Hyperfine'

        return calc

    def create_geom_opt_castep_calculator(self, calc, params={}):
        """Update calculator to contain all the necessary parameters
        for a geometry optimization."""

        # Remove cell constraints if they exist
        calc.cell.cell_constraints = None
        calc.cell.fix_all_cell = True   # Necessary for older CASTEP versions

        calc.param.charge = params.get('charged', False)*1.0

        # Remove symmetry operations if they exist
        calc.cell.symmetry_ops.value = None

        calc.param.task = 'GeometryOptimization'
        calc.param.geom_max_iter = params.get('geom_steps', 30)
        calc.param.geom_force_tol = params.get('geom_force_tol', 0.05)
        calc.param.max_scf_cycles = params.get('max_scc_steps', 30)

        return calc


class CastepError(Exception):
    pass


def castep_write_input(a, folder, calc=None, name=None, script=None):
    """Writes input files for an Atoms object with a Castep
    calculator.

    | Args:
    |   a (ase.Atoms):          Atoms object to write. Can have a Castep
    |                           calculator attached to carry cell/param
    |                           keywords.
    |   folder (str):           Path to save the input files to.
    |   calc (ase.Calculator):  Calculator to attach to Atoms. If
    |                           present, the pre-existent one will
    |                           be ignored.
    |   name (str):             Seedname to save the files with. If not
    |                           given, use the name of the folder.
    |   script (str):           Path to a file containing a submission script
    |                           to copy to the input folder. The script can 
    |                           contain the argument {seedname} in curly braces,
    |                           and it will be appropriately replaced.    
    """

    if name is None:
        name = os.path.split(folder)[-1]  # Same as folder name

    if calc is not None:
        a.set_calculator(calc)

    if not isinstance(a.calc, Castep):
        a = a.copy()
        calc = Castep(atoms=a)
        a.set_calculator(calc)

    io.write(os.path.join(folder, name + '.cell'),
             a, magnetic_moments='initial')
    write_param(os.path.join(folder, name + '.param'),
                a.calc.param, force_write=True)

    if script is not None:
        stxt = open(script).read()
        stxt = stxt.format(seedname=name)
        with open(os.path.join(folder, 'script.sh'), 'w') as sf:
            sf.write(stxt)


def castep_read_input(folder):
    sname = os.path.split(folder)[-1]
    a = io.read(os.path.join(folder, sname + '.castep'))
    return a


def save_muonconf_castep(a, folder, params):
    # Muon mass and gyromagnetic ratio
    mass_block = 'AMU\n{0}       0.1138'
    gamma_block = 'radsectesla\n{0}        851586494.1'

    if isinstance(a.calc, Castep):
        ccalc = a.calc
    else:
        ccalc = Castep()

    ccalc.cell.kpoint_mp_grid.value = list_to_string(params['k_points_grid'])
    ccalc.cell.species_mass = mass_block.format(params['mu_symbol']
                                                ).split('\n')
    ccalc.cell.species_gamma = gamma_block.format(params['mu_symbol']
                                                  ).split('\n')
    ccalc.cell.fix_all_cell = True  # To make sure for older CASTEP versions

    a.set_calculator(ccalc)

    name = os.path.split(folder)[-1]
    io.write(os.path.join(folder, '{0}.cell'.format(name)), a)
    ccalc.atoms = a

    if params['castep_param'] is not None:
        castep_params = yaml.load(open(params['castep_param'], 'r'))
    else:
        castep_params = {}

    # Parameters from .yaml will overwrite parameters from .param
    castep_params['task'] = "GeometryOptimization"
    castep_params['geom_max_iter'] = params['geom_steps']
    castep_params['geom_force_tol'] = params['geom_force_tol']
    castep_params['max_scf_cycles'] = params['max_scc_steps']

    parameter_file = os.path.join(folder, '{0}.param'.format(name))
    yaml.safe_dump(castep_params, open(parameter_file, 'w'),
                   default_flow_style=False)


def parse_castep_bands(infile, header=False):
    """Parse eigenvalues from a CASTEP .bands file. This only works with spin
    components = 1.

    | Args:
    |   infile(str): Directory of bands file.
    |   header(bool, default=False): If true, just return the number of k-points
    |       and eigenvalues. Else, parse and return the band structure.
    | Returns:
    |   n_kpts(int), n_evals(int): Number of k-points and eigenvalues.
    |   bands(Numpy float array, shape:(n_kpts, n_evals)): Energy eigenvalues of
    |       band structure.
    """
    file = open(infile, "r")
    lines = file.readlines()
    n_kpts = int(lines[0].split()[-1])
    n_evals = int(lines[3].split()[-1])
    if header == True:
        return n_kpts, n_evals
    if int(lines[1].split()[-1]) != 1:
        raise ValueError("""Either incorrect file format detected or greater
                            than 1 spin component used (parse_castep_bands
                            only works with 1 spin component.)""")
    # Parse eigenvalues
    bands = np.zeros((n_kpts, n_evals))
    for kpt in range(n_kpts):
        for eval in range(n_evals):
            bands[kpt][eval] = float(lines[11+eval+kpt*(n_evals+2)].strip())
    return bands


def parse_castep_mass_block(mass_block):
    """Parse CASTEP custom species masses, returning a dictionary of masses
    by species, in amu.

    | Args:
    |   mass_block (str):   Content of a species_mass block
    | Returns:
    |   masses (dict):      Dictionary of masses by species symbol
    """

    mass_tokens = [l.split() for l in mass_block.split('\n')]
    custom_masses = {}

    units = {
        'amu': 1,
        'm_e': cnst.m_e/cnst.u,
        'kg': 1.0/cnst.u,
        'g': 1e-3/cnst.u
    }

    # Is the first line a unit?
    u = 1
    if len(mass_tokens) > 0 and len(mass_tokens[0]) == 1:
        try:
            u = units[mass_tokens[0][0].lower()]
        except KeyError:
            raise CastepError('Invalid mass unit in species_mass block')

        mass_tokens.pop(0)

    for tk in mass_tokens:
        try:
            custom_masses[tk[0]] = float(tk[1])*u
        except (ValueError, IndexError):
            raise CastepError('Invalid line in species_mass block')

    return custom_masses


def parse_castep_masses(cell):
    """Parse CASTEP custom species masses, returning an array of all atom 
    masses in .cell file with corrected custom masses.

    | Args:
    |   cell(ASE Atoms object): Atoms object containing relevant .cell file
    | Returns:
    |   masses(Numpy float array, shape(no. of atoms)): Correct masses of all
    |       atoms in cell file.
    """
    mass_block = cell.calc.cell.species_mass.value
    if mass_block is None:
        return cell.get_masses()

    custom_masses = parse_castep_mass_block(mass_block)

    masses = cell.get_masses()
    elems = cell.get_chemical_symbols()
    elems = cell.arrays.get('castep_custom_species', elems)

    masses = [custom_masses.get(elems[i], m) for i, m in enumerate(masses)]

    cell.set_masses(masses)

    return masses


def parse_castep_gamma_block(gamma_block):
    """Parse CASTEP custom species gyromagnetic ratios, returning a 
    dictionary of gyromagnetic ratios by species, in radsectesla.

    | Args:
    |   gamma_block (str):   Content of a species_gamma block
    | Returns:
    |   gammas (dict):      Dictionary of gyromagnetic ratios by species symbol
    """

    gamma_tokens = [l.split() for l in gamma_block.split('\n')]
    custom_gammas = {}

    units = {
        'agr': cnst.e/cnst.m_e,
        'radsectesla': 1,
        'mhztesla': 0.5e-6/np.pi,
    }

    # Is the first line a unit?
    u = 1
    if len(gamma_tokens) > 0 and len(gamma_tokens[0]) == 1:
        try:
            u = units[gamma_tokens[0][0].lower()]
        except KeyError:
            raise CastepError('Invalid gamma unit in species_gamma block')

        gamma_tokens.pop(0)

    for tk in gamma_tokens:
        try:
            custom_gammas[tk[0]] = float(tk[1])*u
        except (ValueError, IndexError):
            raise CastepError('Invalid line in species_gamma block')

    return custom_gammas

def parse_castep_ppots(cfile):

    clines = open(cfile).readlines()

    # Find pseudopotential blocks
    ppot_heads = filter(lambda x: 'Pseudopotential Report' in x[1],
                        enumerate(clines))
    ppot_blocks_raw = []

    for pph in ppot_heads:
        i, _ = pph
        for j, l in enumerate(clines[i:]):
            if 'Author:' in l:
                break
        ppot_blocks_raw.append(clines[i:i+j])

    # Now on to actually parse them
    ppot_blocks = {}

    el_re = re.compile(r'Element:\s+([a-zA-Z]{1,2})\s+'
                       r'Ionic charge:\s+([0-9.]+)')
    rc_re = re.compile(r'(?:[0-9]+|loc)\s+[0-9]\s+[\-0-9.]+\s+([0-9.]+)')
    bohr = cnst.physical_constants['Bohr radius'][0]*1e10

    for ppb in ppot_blocks_raw:
        el = None
        q = None
        rcmin = np.inf
        for l in ppb:
            el_m = el_re.search(l)
            if el_m is not None:
                el, q = el_m.groups()
                q = float(q)
                continue
            rc_m = rc_re.search(l)
            if rc_m is not None:
                rc = float(rc_m.groups()[0])*bohr
                rcmin = min(rc, rcmin)
        ppot_blocks[el] = (q, rcmin)

    return ppot_blocks


def parse_final_energy(infile):
    """
    Parse final energy from .castep file

    | Args:
    |   infile (str): Directory of .castep file
    |
    | Returns:
    |   E (float): Value of final energy
    """
    E = None
    for l in open(infile).readlines():
        if "Final energy" in l:
            try:
                E = float(l.split()[3])
            except ValueError:
                raise RuntimeError(
                    "Corrupt .castep file found: {0}".format(infile))
    return E


def add_to_castep_block(cblock, symbol, value, blocktype='mass'):
    """Add a pair of the form:
        symbol  value
       to a given castep block cblock, given the type.
    """

    parser = {
        'mass': parse_castep_mass_block,
        'gamma': parse_castep_gamma_block
    }[blocktype]

    if cblock is None:
        values = {}
    else:
        values = parser(cblock)
    # Assign the muon mass
    values[symbol] = value

    cblock = {
        'mass': 'AMU',
        'gamma': 'radsectesla'
    }[blocktype] + '\n'
    for k, v in values.items():
        cblock += '{0} {1}\n'.format(k, v)

    return cblock
