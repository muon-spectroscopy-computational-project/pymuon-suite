import glob
import os
import re
from copy import deepcopy

import numpy as np
import scipy.constants as cnst

from ase import io
from ase.calculators.castep import Castep
from ase.io.castep import read_param, write_param
from ase.io.magres import read_magres

from pymuonsuite import constants
from pymuonsuite.io.readwrite import ReadWrite
from pymuonsuite.optional import requireEuphonicQPM
from pymuonsuite.utils import list_to_string
from soprano.utils import customize_warnings, seedname, silence_stdio

customize_warnings()


class ReadWriteCastep(ReadWrite):
    def __init__(self, params={}, script=None, calc=None):
        """
        |   params (dict)           Contains muon symbol, parameter file,
        |                           k_points_grid.
        |   script (str):           Path to a file containing a submission
        |                           script to copy to the input folder. The
        |                           script can contain the argument
        |                           {seedname} in curly braces, and it will
        |                           be appropriately replaced.
        |   calc (ase.Calculator):  Calculator to attach to Atoms. If
        |                           present, the pre-existent one will
        |                           be ignored.
        """
        self.params = self._validate_params(params)
        self.script = script
        self._calc = calc
        if calc is not None and params != {}:
            self._create_calculator()

    def _validate_params(self, params):
        if not (isinstance(params, dict)):
            raise ValueError("params should be a dict, not ", type(params))
            return
        else:
            return params

    def set_params(self, params):
        """
        |   params (dict)           Contains muon symbol, parameter file,
        |                           k_points_grid.
        """
        self.params = self._validate_params(params)
        # if the params have been changed, the calc has to be remade
        # from scratch:
        self._calc = None
        self._create_calculator()

    def read(self, folder, sname=None, read_magres=False, read_phonons=False):

        """Reads Castep output files.

        | Args:
        |   folder (str):           Path to folder from which to read files.
        |   sname (str):            Seedname to save the files with. If not
        |                           given, use the name of the folder.
        """
        atoms = self._read_castep(folder, sname)
        if read_magres:
            self._read_castep_hyperfine_magres(atoms, folder, sname)
        if read_phonons:
            self._read_castep_gamma_phonons(atoms, folder, sname)
        return atoms

    def _read_castep(self, folder, sname=None):
        try:
            if sname is not None:
                cfile = os.path.join(folder, sname + ".castep")
            else:
                cfile = glob.glob(os.path.join(folder, "*.castep"))[0]
                sname = seedname(cfile)
            with silence_stdio():
                atoms = io.read(cfile)
            atoms.info["name"] = sname
            return atoms

        except IndexError:
            raise IOError(
                "ERROR: No .castep files found in {}.".format(os.path.abspath(folder))
            )
        except OSError as e:
            raise IOError("ERROR: {}".format(e))
        except (
            io.formats.UnknownFileTypeError,
            ValueError,
            TypeError,
            Exception,
        ) as e:
            raise IOError(
                "ERROR: Invalid file: {file}, due to error: {error}".format(
                    file=sname + ".castep", error=e
                )
            )

    def _read_castep_hyperfine_magres(self, atoms, folder, sname=None):
        try:
            if sname is not None:
                mfile = os.path.join(folder, sname + ".magres")
            else:
                mfile = glob.glob(os.path.join(folder, "*.magres"))[0]
            m = parse_hyperfine_magres(mfile)
            atoms.arrays.update(m.arrays)
        except (IndexError, OSError):
            raise IOError(
                "No .magres files found in {}.".format(os.path.abspath(folder))
            )

    @requireEuphonicQPM("QpointPhononModes")
    def _read_castep_gamma_phonons(
        self, atoms, folder, sname=None, QpointPhononModes=None
    ):
        """Parse CASTEP phonon data into a casteppy object,
        and return eigenvalues and eigenvectors at the gamma point.
        """

        # Parse CASTEP phonon data into casteppy object
        try:
            if sname is not None:
                pd = QpointPhononModes.from_castep(
                    os.path.join(folder, sname + ".phonon")
                )
            else:
                pd = QpointPhononModes.from_castep(
                    glob.glob(os.path.join(folder, "*.phonon"))[0]
                )
                sname = seedname(glob.glob(os.path.join(folder, "*.phonon"))[0])
            # Convert frequencies back to cm-1
            pd.frequencies_unit = "1/cm"
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
                raise CastepError(
                    "Could not find gamma point phonons in" " CASTEP phonon file"
                )

            atoms.info["ph_evals"] = evals[gamma_i]
            atoms.info["ph_evecs"] = evecs[gamma_i]

        except IndexError:
            raise IOError(
                "No .phonon files found in {}.".format(os.path.abspath(folder))
            )
        except (OSError, IOError) as e:
            raise IOError("ERROR: {}".format(e))
        except Exception as e:
            raise IOError(
                "ERROR: Could not read {file} due to error: {e}".format(
                    file=sname + ".phonon", e=e
                )
            )

    def write(self, a, folder, sname=None, calc_type="GEOM_OPT"):
        """Writes input files for an Atoms object with a Castep
        calculator.

        | Args:
        |   a (ase.Atoms):          Atoms object to write. Can have a Castep
        |                           calculator attached to carry cell/param
        |                           keywords.
        |   folder (str):           Path to save the input files to.
        |   sname (str):            Seedname to save the files with. If not
        |                           given, use the name of the folder.
        |   calc_type (str):        Castep task which will be performed:
        |                           "GEOM_OPT" or "MAGRES"
        """
        if calc_type == "GEOM_OPT" or calc_type == "MAGRES":
            if sname is None:
                sname = os.path.split(folder)[-1]  # Same as folder name

            self._calc = deepcopy(self._calc)

            # We only use the calculator attached to the atoms object if a calc
            # has not been set when initialising the ReadWrite object OR we
            # have not called write() and made a calculator before.

            if self._calc is None:
                if isinstance(a.calc, Castep):
                    self._calc = deepcopy(a.calc)
                self._create_calculator(calc_type=calc_type)
            else:
                self._update_calculator(calc_type)
            a.calc = self._calc
            with silence_stdio():
                io.write(
                    os.path.join(folder, sname + ".cell"),
                    a,
                    magnetic_moments="initial",
                )
            write_param(
                os.path.join(folder, sname + ".param"),
                a.calc.param,
                force_write=True,
            )

            if self.script is not None:
                stxt = open(self.script).read()
                stxt = stxt.format(seedname=sname)
                with open(os.path.join(folder, "script.sh"), "w", newline="\n") as sf:
                    sf.write(stxt)
        else:
            raise (
                NotImplementedError(
                    "Calculation type {} is not implemented."
                    " Please choose 'GEOM_OPT' or 'MAGRES'".format(calc_type)
                )
            )

    def _create_calculator(self, calc_type=None):
        with silence_stdio():
            if self._calc is not None and isinstance(self._calc, Castep):
                calc = deepcopy(self._calc)
            else:
                calc = Castep()

        mu_symbol = self.params.get("mu_symbol", "H:mu")
        particle_mass = self.params.get("particle_mass_amu", constants.m_mu_amu)

        # Start by ensuring that the muon mass and gyromagnetic ratios are
        # included
        gamma_block = calc.cell.species_gamma.value
        if gamma_block is None:
            calc.cell.species_gamma = add_to_castep_block(
                gamma_block, mu_symbol, constants.m_gamma, "gamma"
            )

            mass_block = calc.cell.species_mass.value
            calc.cell.species_mass = add_to_castep_block(
                mass_block, mu_symbol, particle_mass, "mass"
            )

        # Now assign the k-points
        k_points_param = self.params.get("k_points_grid")

        if k_points_param is not None:
            calc.cell.kpoint_mp_grid = list_to_string(k_points_param)
        else:
            if calc.cell.kpoint_mp_grid is None:
                calc.cell.kpoint_mp_grid = list_to_string([1, 1, 1])

        # Read the parameters
        pfile = self.params.get("castep_param", None)
        if pfile is not None:
            with silence_stdio():
                calc.param = read_param(self.params["castep_param"]).param

        self._calc = calc

        if calc_type == "MAGRES":
            calc = self._create_hfine_castep_calculator()
        elif calc_type == "GEOM_OPT":
            calc = self._create_geom_opt_castep_calculator()

        return self._calc

    def _update_calculator(self, calc_type):
        if calc_type == "MAGRES":
            # check if our calculator is already set up for Magres
            # if it is, we don't need to modify the calc.
            if not self._calc.param.task == "Magres":
                self._create_hfine_castep_calculator()
        elif calc_type == "GEOM_OPT":
            # check if our calculator is already set up for geom opt
            # if it is, we don't need to modify the calc.
            if not self._calc.param.task == "GeometryOptimization":
                self._create_geom_opt_castep_calculator()
        return self._calc

    def _create_hfine_castep_calculator(self):
        """Update calculator to contain all the necessary parameters
        for a hyperfine calculation."""

        # Remove settings for geom_opt calculator:
        self._calc.param.geom_max_iter = None
        self._calc.param.geom_force_tol = None
        self._calc.param.max_scf_cycles = None
        self._calc.param.write_cell_structure = None
        self._calc.param.charge = None
        self._calc.cell.fix_all_cell = None

        pfile = self.params.get("castep_param", None)
        if pfile is not None:
            with silence_stdio():
                self._calc.param = read_param(self.params["castep_param"]).param

        self._calc.param.task = "Magres"
        self._calc.param.magres_task = "Hyperfine"

        return self._calc

    def _create_geom_opt_castep_calculator(self):
        """Update calculator to contain all the necessary parameters
        for a geometry optimization."""

        # Remove cell constraints if they exist
        self._calc.cell.cell_constraints = None
        self._calc.cell.fix_all_cell = True

        self._calc.param.task = "GeometryOptimization"

        # Remove symmetry operations if they exist
        self._calc.cell.symmetry_ops.value = None

        # If the following parameters are set in the params dict we take
        # their values from there.
        # Otherwise, we take their values from the calculator that has
        # been provided.
        # If neither of these have been set, we use the default values.

        charge_param = self.params.get("charged")

        if charge_param is not None:
            self._calc.param.charge = charge_param * 1.0
        else:
            if self._calc.param.charge is None:
                self._calc.param.charge = False * 1.0

        geom_steps_param = self.params.get("geom_steps")

        if geom_steps_param is not None:
            self._calc.param.geom_max_iter = geom_steps_param
        else:
            if self._calc.param.geom_max_iter.value is None:
                self._calc.param.geom_max_iter = 30

        geom_force_tol_param = self.params.get("geom_force_tol")

        if geom_force_tol_param is not None:
            self._calc.param.geom_force_tol = geom_force_tol_param
        else:
            if self._calc.param.geom_force_tol.value is None:
                self._calc.param.geom_force_tol = 0.05

        max_scf_cycles_param = self.params.get("max_scc_steps")

        if max_scf_cycles_param is not None:
            self._calc.param.max_scf_cycles.value = max_scf_cycles_param

        else:
            if self._calc.param.max_scf_cycles.value is None:
                self._calc.param.max_scf_cycles = 30

        self._calc.param.write_cell_structure = True  # outputs -out.cell file

        # Remove settings for magres calculator:
        self._calc.param.magres_task = None

        return self._calc


class CastepError(Exception):
    pass


def parse_castep_bands(infile, header=False):
    """Parse eigenvalues from a CASTEP .bands file. This only works with spin
    components = 1.

    | Args:
    |   infile(str): Directory of bands file.
    |   header(bool, default=False): If true, just return the number of
    |   k-points and eigenvalues. Else, parse and return the band structure.
    | Returns:
    |   n_kpts(int), n_evals(int): Number of k-points and eigenvalues.
    |   bands(Numpy float array, shape:(n_kpts, n_evals)): Energy eigenvalues
    |   of band structure.
    """
    file = open(infile, "r")
    lines = file.readlines()
    n_kpts = int(lines[0].split()[-1])
    n_evals = int(lines[3].split()[-1])
    if header is True:
        return n_kpts, n_evals
    if int(lines[1].split()[-1]) != 1:
        raise ValueError(
            """Either incorrect file format detected or greater
                            than 1 spin component used (parse_castep_bands
                            only works with 1 spin component.)"""
        )
    # Parse eigenvalues
    bands = np.zeros((n_kpts, n_evals))
    for kpt in range(n_kpts):
        for eval in range(n_evals):
            bands[kpt][eval] = float(lines[11 + eval + kpt * (n_evals + 2)].strip())
    return bands


def parse_castep_mass_block(mass_block):
    """Parse CASTEP custom species masses, returning a dictionary of masses
    by species, in amu.

    | Args:
    |   mass_block (str):   Content of a species_mass block
    | Returns:
    |   masses (dict):      Dictionary of masses by species symbol
    """

    mass_tokens = [lm.split() for lm in mass_block.split("\n")]
    custom_masses = {}

    units = {
        "amu": 1,
        "m_e": cnst.m_e / cnst.u,
        "kg": 1.0 / cnst.u,
        "g": 1e-3 / cnst.u,
    }

    # Is the first line a unit?
    u = 1
    if len(mass_tokens) > 0 and len(mass_tokens[0]) == 1:
        try:
            u = units[mass_tokens[0][0].lower()]
        except KeyError:
            raise CastepError("Invalid mass unit in species_mass block")

        mass_tokens.pop(0)

    for tk in mass_tokens:
        try:
            custom_masses[tk[0]] = float(tk[1]) * u
        except (ValueError, IndexError):
            raise CastepError("Invalid line in species_mass block")

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
    elems = cell.arrays.get("castep_custom_species", elems)

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

    gamma_tokens = [lg.split() for lg in gamma_block.split("\n")]
    custom_gammas = {}

    units = {
        "agr": cnst.e / cnst.m_e,
        "radsectesla": 1,
        "mhztesla": 0.5e-6 / np.pi,
    }

    # Is the first line a unit?
    u = 1
    if len(gamma_tokens) > 0 and len(gamma_tokens[0]) == 1:
        try:
            u = units[gamma_tokens[0][0].lower()]
        except KeyError:
            raise CastepError("Invalid gamma unit in species_gamma block")

        gamma_tokens.pop(0)

    for tk in gamma_tokens:
        try:
            custom_gammas[tk[0]] = float(tk[1]) * u
        except (ValueError, IndexError):
            raise CastepError("Invalid line in species_gamma block")

    return custom_gammas


def parse_castep_ppots(cfile):

    clines = open(cfile).readlines()

    # Find pseudopotential blocks
    ppot_heads = filter(lambda x: "Pseudopotential Report" in x[1], enumerate(clines))
    ppot_blocks_raw = []

    for pph in ppot_heads:
        i, _ = pph
        for j, l in enumerate(clines[i:]):
            if "Author:" in l:
                break
        ppot_blocks_raw.append(clines[i : i + j])

    # Now on to actually parse them
    ppot_blocks = {}

    el_re = re.compile(r"Element:\s+([a-zA-Z]{1,2})\s+" r"Ionic charge:\s+([0-9.]+)")
    rc_re = re.compile(r"(?:[0-9]+|loc)\s+[0-9]\s+[\-0-9.]+\s+([0-9.]+)")
    bohr = cnst.physical_constants["Bohr radius"][0] * 1e10

    for ppb in ppot_blocks_raw:
        el = None
        q = None
        rcmin = np.inf
        for lp in ppb:
            el_m = el_re.search(lp)
            if el_m is not None:
                el, q = el_m.groups()
                q = float(q)
                continue
            rc_m = rc_re.search(lp)
            if rc_m is not None:
                rc = float(rc_m.groups()[0]) * bohr
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
    for lf in open(infile).readlines():
        if "Final energy" in lf:
            try:
                E = float(lf.split()[3])
            except ValueError:
                raise RuntimeError("Corrupt .castep file found: {0}".format(infile))
    return E


def add_to_castep_block(cblock, symbol, value, blocktype="mass"):
    """Add a pair of the form:
     symbol  value
    to a given castep block cblock, given the type.
    """

    parser = {
        "mass": parse_castep_mass_block,
        "gamma": parse_castep_gamma_block,
    }[blocktype]

    if cblock is None:
        values = {}
    else:
        values = parser(cblock)
    # Assign the muon mass
    values[symbol] = value

    cblock = {"mass": "AMU", "gamma": "radsectesla"}[blocktype] + "\n"
    for k, v in values.items():
        cblock += "{0} {1}\n".format(k, v)

    return cblock


def parse_hyperfine_magres(infile):
    """
    Parse hyperfine values from .magres file

    | Args:
    |   infile (str): Directory of .magres file
    |
    | Returns:
    |   mgr (ASE Magres object): Object containing .magres hyperfine data
    """

    file = open(infile, "r")
    # First, `simply parse the magres file via ASE
    mgr = read_magres(file, True)

    # Now go for the magres_old block

    if "magresblock_magres_old" not in mgr.info:
        raise RuntimeError(".magres file has no hyperfine information")

    hfine = parse_hyperfine_oldblock(mgr.info["magresblock_magres_old"])

    labels, indices = mgr.get_array("labels"), mgr.get_array("indices")

    hfine_array = []
    for lb, i in zip(labels, indices):
        hfine_array.append(hfine[lb][i])

    mgr.new_array("hyperfine", np.array(hfine_array))

    return mgr


def parse_hyperfine_oldblock(block):
    """
    Parse a magres_old block into a dictionary

    | Args:
    |   block (str): magres_old block
    |
    | Returns:
    |   hfine_dict (dict{"species" (str):tensor (int[3][3])}):
    |                         Dictionary containing hyperfine data
    """

    hfine_dict = {}

    sp = None
    n = None
    tens = None
    block_lines = block.split("\n")
    for i, l in enumerate(block_lines):
        if "Atom:" in l:
            # Get the species and index
            _, sp, n = l.split()
            n = int(n)
        if "TOTAL tensor" in l:
            tens = np.array(
                [[float(x) for x in row.split()] for row in block_lines[i + 2 : i + 5]]
            )
            # And append
            if sp is None:
                raise RuntimeError("Invalid block in magres hyperfine file")
            if sp not in hfine_dict:
                hfine_dict[sp] = {}
            hfine_dict[sp][n] = tens

    return hfine_dict
