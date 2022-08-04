"""
schemas.py

Python schemas for YAML input files for various scripts
"""


import yaml
import numpy as np
import warnings
from schema import Optional, Schema, SchemaError
from scipy.constants import physical_constants as pcnst
from soprano.utils import customize_warnings

customize_warnings()


def validate_matrix_shape(x):
    x = np.array(x)
    return x.shape == (1,) or x.shape == (3,) or x.shape == (9,) or x.shape == (3, 3)


def validate_supercell(value):
    return (
        value is None
        or isinstance(value, int)
        or (validate_matrix_shape(value) and np.array(value).dtype == int)
    )


def validate_all_of(*options, case_sensitive=True):
    def validator(values):
        if case_sensitive:
            return all([x.strip() in options for x in values.split(",")])
        else:
            return all([x.strip().lower() in options for x in values.split(",")])

    return validator


def validate_str(s):
    # Validates a string in a Python 2 and 3 compatible way
    try:
        return isinstance(s, basestring)
    except NameError:
        # It must be Python 3!
        return isinstance(s, str)


def validate_str_list(value):
    return all(map(validate_str, value))


def validate_bool(value):
    return str(value).strip().lower() in ("true", "t", "false", "f")


def validate_int3(value):
    v = np.array(value)
    return v.shape == (3,) and v.dtype == int


def validate_int_array(value):
    v = np.array(value)
    return v.dtype == int


def validate_vec3(value):
    try:
        v = np.array(value).astype(float)
    except ValueError:
        return False
    return v.shape == (3,)


def validate_save_min(value):
    warnings.warn(
        "The 'clustering_save_min' option is deprecated. "
        "It has been replaced with two options: "
        "'clustering_save_type' and 'clustering_save_format."
        "'clustering_save_type' : structures/input - for selecting "
        "whether to save structure files or input files to another "
        "calculator. "
        "'clustering_save_format': if structures, this "
        "takes an extension (cif, xyz, etc.). If input, takes the "
        "name of a program (castep, dftb+, etc.)"
    )
    return isinstance(value, bool)


def load_input_file(fname, param_schema, merge=None):
    """Load a given input YAML file and validate it with a schema."""

    if merge is None:
        with open(fname, "r") as params_file:
            params = yaml.safe_load(params_file)
    else:
        try:
            param_schema.validate(merge)
        except SchemaError as e:
            message = "Invalid merge params passed to" " load_input_file\n{0}".format(e)
            raise RuntimeError(message)
        with open(fname, "r") as params_file:
            new_params = yaml.safe_load(params_file)
        params = dict(merge)
        params.update(new_params)

    if params is None:
        params = {}  # Fix in case the yaml file is empty

    try:
        params = param_schema.validate(params)
    except SchemaError as e:
        message = "Invalid input file {0}\n{1}".format(fname, e)
        raise RuntimeError(message)

    return params


# Parameter file schemas and defaults
MuAirssSchema = Schema(
    {
        # Name to call the folder for containing each structure. This name will
        # be postfixed with a unique number. e.g. struct_001
        Optional("name", default="struct"): validate_str,
        # Calculator to generate structure files for. Must be a comma seperated
        # list of values. Currently supported calculators are CASTEP, DFTB+ and
        # UEP. Can also pass all as an option to generate files for all
        # calculators.
        Optional("calculator", default="dftb+"): validate_all_of(
            "castep", "dftb+", "uep", "all", case_sensitive=False
        ),
        # Command to use to run CASTEP.
        Optional("castep_command", default="castep.serial"): validate_str,
        # Command to use to run DFTB+.
        Optional("dftb_command", default="dftb+"): validate_str,
        # Path to script file to copy in all folders
        Optional("script_file", default=None): validate_str,
        # File path to the CASTEP parameter file.
        Optional("castep_param", default=None): validate_str,
        # The parameter set to use for DFTB+
        Optional("dftb_set", default="3ob-3-1"): validate_all_of("3ob-3-1", "pbc-0-3"),
        # Additional optional json files to activate for DFTBArgs
        Optional("dftb_optionals", default=[]): validate_str_list,
        # Whether to turn on periodic boundary conditions in DFTB+
        Optional("dftb_pbc", default=True): bool,
        # Charge density file for UEP
        Optional("uep_chden", default=""): validate_str,
        # Gaussian Width factor for UEP
        Optional("uep_gw_factor", default=5.0): float,
        # Radius to use when generating muon sites with the possion disk algorithm.
        Optional("poisson_r", default=0.8): float,
        # Van der Waals radius to use when generating muon sites. Default is 0.5.
        Optional("vdw_scale", default=0.5): float,
        # Whether to make the muon charged (naked muon instead of muonium)
        Optional("charged", default=False): bool,
        # Supercell size and shape to use. This can either be a single int, a list
        # of three integers or a 3x3 matrix of integers. For a single number a
        # diagonal matrix will be generated with the integer repeated on the
        # diagonals. For a list of three numbers a diagonal matrix will be
        # generated where the diagonal elements are set to the list. A matrix will
        # be used directly as is. Default is a 3x3 identity matrix.
        Optional("supercell", default=1): validate_supercell,
        # List of three integer k-points. Default is [1,1,1].
        Optional("k_points_grid", default=np.ones(3).astype(int)): validate_int3,
        # Name to call the output folder used to store the input files
        # that muon-airss generates.
        Optional("out_folder", default="./muon-airss-out"): validate_str,
        # Save structure file for each optimised
        # struct + muon in .xyz format when running uep
        Optional("uep_save_structs", default=True): bool,
        # The symbol to use for the muon when writing out the castep custom
        # species.
        Optional("mu_symbol", default="H:mu"): validate_str,
        # Maximum number of geometry optimisation steps
        Optional("geom_steps", default=None): int,
        # Tolerance on geometry optimisation in units of eV/AA.
        Optional("geom_force_tol", default=None): float,
        # Max number of SCC steps to perform before giving up. Default is
        # 200 which is also the default for DFTB+.
        Optional("max_scc_steps", default=None): int,
        # Clustering method to use
        Optional("clustering_method", default="hier"): validate_all_of(
            "hier", "kmeans"
        ),
        # t parameter for hierarchical clustering
        Optional("clustering_hier_t", default=0.3): float,
        # Number of clusters for k-means clustering
        Optional("clustering_kmeans_k", default=4): int,
        # This option is deprecated:
        # Whether to save the minimum energy structures for each cluster
        Optional("clustering_save_min", default=False): validate_save_min,
        # Whether to save the minimum energy structures for each cluster,
        # or input files to another calculator
        Optional("clustering_save_type", default=None): validate_all_of(
            "structures", "input", None
        ),
        # Format to use to save the minimum energy structures for each cluster
        Optional("clustering_save_format", default=None): validate_str,
        # Name to call the folder used to store the input files
        # for castep or dftb+ calculations and/or the minimum energy
        # structures for each cluster
        Optional("clustering_save_folder", default=None): validate_str,
        # Save a file with all muon positions in one
        Optional("allpos_filename", default=None): validate_str,
        # Random seed for generation
        Optional("random_seed", default=None): int,
    }
)

# Parameter file schema and defaults
MuonHarmonicSchema = Schema(
    {
        # Method used to calculate thermal average
        Optional("method", default="independent"): validate_all_of(
            "independent", "montecarlo"
        ),
        # Index of muon in cell
        Optional("mu_index", default=-1): int,
        # If using Castep custom species, custom species of muon (supersedes index
        # if present in cell)
        Optional("mu_symbol", default="H:mu"): validate_str,
        # Number of grid points to use on each phonon mode or pairs of
        # thermal lines
        Optional("grid_n", default=20): int,
        # Number of sigmas to sample in the harmonic approximation
        Optional("sigma_n", default=3): float,
        # List of three integer k-points for both phonon and hyperfine
        # calculations. Default is [1,1,1].
        Optional("k_points_grid", default=np.ones(3).astype(int)): validate_int3,
        # Property to be calculated, currently accepted value is only 'hyperfine'
        # (hyperfine coupling tensors)
        Optional("avgprop", default="hyperfine"): validate_all_of(
            "hyperfine", "charge"
        ),
        # Calculator to use for property
        Optional("calculator", default="castep"): validate_all_of("castep", "dftb+"),
        # Write a 'collective' file with all displaced positions in one
        Optional("write_allconf", default=False): validate_bool,
        # Source file for phonon modes
        Optional("phonon_source_file", default=None): validate_str,
        # Type of source file for phonon modes
        Optional("phonon_source_type", default="castep"): validate_all_of(
            "castep", "dftb+"
        ),
        # Temperature for displacement generation
        Optional("displace_T", default=0): float,
        # Temperature for averaging
        Optional("average_T", default=None): float,
        # Calculation parameters
        # Path to script file to copy in all folders
        Optional("script_file", default=None): validate_str,
        # Path of parameter file which can be copied into folders with displaced
        # cell files for convenience
        Optional("castep_param", default=None): validate_str,
        # Whether to turn on periodic boundary conditions in DFTB+
        Optional("dftb_pbc", default=True): bool,
        # If using DFTB+, which parametrization to use
        Optional("dftb_set", default="3ob-3-1"): validate_all_of("3ob-3-1", "pbc-0-3"),
        # Output files
        Optional("average_file", default="averages.dat"): validate_str,
        # Random seed for generation
        Optional("random_seed", default=None): int,
    }
)

AsePhononsSchema = Schema(
    {
        # Name
        Optional("name", default=None): validate_str,
        # Phonon k-points
        Optional("phonon_kpoint_grid", default=[1, 1, 1]): validate_int3,
        # K-points used for DFTB+ calculation
        Optional("kpoint_grid", default=[1, 1, 1]): validate_int3,
        # Force tolerance for optimisation
        Optional("force_tol", default=0.01): float,
        # Which parametrization to use
        Optional("dftb_set", default="3ob-3-1"): validate_all_of("3ob-3-1", "pbc-0-3"),
        # Whether to turn on periodic boundary conditions
        Optional("pbc", default=True): validate_bool,
        # Force clean existing phonon files of the same name
        Optional("force_clean", default=False): validate_bool,
    }
)

# Shared schema for all UEP calculations
UEPSchema = Schema(
    {
        # Path from which to load the charge density
        Optional("chden_path", default=""): validate_str,
        # Seedname for the charge density calculation
        Optional("chden_seed", default=None): validate_str,
        # Gaussian Width factor for ionic potential
        Optional("gw_factor", default=5.0): float,
    }
)

# UEP muon position optimisation
UEPOptSchema = UEPSchema.schema.copy()
UEPOptSchema.update(
    {
        # Starting position for muon (absolute coordinates)
        Optional("mu_pos", default=[0.0, 0.0, 0.0]): validate_vec3,
        # Maximum number of geometry optimisation steps
        Optional("geom_steps", default=30): int,
        # Tolerance on optimisation
        Optional("opt_tol", default=1e-5): float,
        # Optimisation method
        Optional("opt_method", default="trust-exact"): validate_str,
        # Particle mass (in kg)
        Optional("particle_mass", default=pcnst["muon mass"][0]): float,
        # Save pickled output
        Optional("save_pickle", default=True): bool,
        # Save structure file for each optimised
        # struct + muon in .xyz format
        Optional("save_structs", default=True): bool,
    }
)
UEPOptSchema = Schema(UEPOptSchema)

# UEP plotting
UEPPlotSchema = UEPSchema.schema.copy()
UEPPlotSchema.update(
    {
        # Specifications for paths.
        # Possible formats:
        # - [[crystallographic direction], [starting point], length,
        #   number of points]
        # - [[starting point], [end point], number of points],
        # - [starting atom, end atom, number of points]
        Optional("line_plots", default=[]): list,
        # Specifications for planes.
        # Possible formats:
        # - [[corner 1], [corner 2], [corner 3],
        #     points along width, points along height]
        # - [corner atom 1, corner atom 2, corner atom 3,
        #    points along width, points along height]
        Optional("plane_plots", default=[]): list,
    }
)
UEPPlotSchema = Schema(UEPPlotSchema)
