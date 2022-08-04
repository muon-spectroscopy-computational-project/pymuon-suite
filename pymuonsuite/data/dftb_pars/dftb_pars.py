import os
import json
import glob
from copy import deepcopy

# Upon loading, print a message for the user
_references_msg = """
This calculation makes use of the DFTB parametrisations found at

    https://www.dftb.org/

All data is licensed under the Creative Commons Attribution-ShareAlike 4.0
International License. For specific data sets remember to cite the following
papers, by element:

3-ob-3-1
    [JCTC2013]  J. Chem. Theory Comput., 2013, 9, 338-354.      (O, N, C, H)
    [JCTC2014]  J. Chem. Theory Comput., 2014, 10, 1518-1537.   (P,S-*)
    [JCTC2015-1]    J. Phys. Chem. B, 2015, 119, 1062-1082.     (Mg,Zn-*)
    [JCTC2015-2]    J. Chem. Theory Comput., 2015, 11, 332-342. (Na,F,K,Ca,Cl,Br,I-*)

pbc-0-3
    [SiC]   E. Rauls, R. Gutierrez, J. Elsner, and Th. Frauenheim,
                Sol. State Comm. 111, 459 (1999).                           (Si-C)
    [SiO]   C. Koehler, Z. Hajnal, P. Deak, Th. Frauenheim, S. Suhai,
                Phys. Rev. B 64, 085333 (2001).                             (Si-O)
    [Silicon]   A. Sieck, Th. Frauenheim, and K. A. Jackson,
                    Phys. Stat. Sol. (b) 240, 537 (2003).                   (Si)
    [Fluorine]  C. Koehler and Th. Frauenheim,
                    Surf. Sci. 600, 453 (2006).                             (F)
    [Iron]  C. Koehler, G. Seifert and Th. Frauenheim,
                Chem. Phys. 309, 23 (2005).                                 (Fe)
    [SiSi]  A. Sieck,
                PhD. Thesis, University of Paderborn, 2000.                 (Si-Si)
"""


def print_references():
    print(_references_msg)


def get_license():
    return open(os.path.join(os.path.split(__file__)[0], "LICENSE")).read()


print_references()  # Print whenever it gets loaded


def parse_params(dir):
    name = dir.split(os.sep)[-2]
    # Start by defining the Slater Koster path:
    args = {
        "Hamiltonian_SlaterKosterFiles_Prefix": os.path.abspath(
            os.path.join(dir, name, "")
        )
        + os.sep
    }

    # Try loading any additional arguments
    try:
        with open(os.path.join(dir, "args.json")) as args_file:
            args.update(json.load(args_file))
    except IOError:
        pass

    return args


parameter_sets = {
    d.split(os.sep)[-2]: parse_params(d)
    for d in glob.glob(os.path.join(os.path.dirname(__file__), "*/"))
}


class DFTBArgs(object):
    """DFTBArgs

    Class generating automatically arguments for an ASE DFTB+ calculator.
    """

    def __init__(self, name):
        """Initialise a DFTBArgs object

        Initialise a DFTBArgs object given the name of a parametrisation set
        of choice.

        Arguments:
            name {str} -- Name of chosen parametrisation set. Currently
                          acceptable values are 3ob-3-1, pbc-0-3.
        """
        self._name = name
        self._args = parameter_sets[name]
        self._path = os.path.join(os.path.dirname(__file__), name)

        self._optional = [
            f
            for f in glob.glob(os.path.join(self._path, "*.json"))
            if os.path.basename(f) != "args.json"
        ]

        self._optdict = {os.path.basename(f): False for f in self._optional}

    @property
    def name(self):
        return self._name

    @property
    def args(self):

        args = deepcopy(self._args)

        for name, value in self._optdict.items():
            if value:
                with open(os.path.join(self._path, name)) as json_file:
                    args.update(json.load(json_file))

        return args

    @property
    def path(self):
        return self._path

    def set_optional(self, name=None, value=False):
        """Set optional properties

        Set optional argument files for the given parametrization set on or
        off.
        If called without any argument, prints out the optional arguments
        available.

        Keyword Arguments:
            name {str} -- Optional argument file to set (default: {None})
            value {bool} -- If True, include the optional arguments
                            (default: {False})

        Raises:
            KeyError -- The optional file is not available
        """
        if name is None:
            print("Optional files available:")
            print("\n".join(self._optdict.keys()))
        else:
            if name not in self._optdict:
                raise KeyError("Optional file {0} not available".format(name))
            self._optdict[name] = value

    @staticmethod
    def list():
        """Print list of available parameter set names"""
        print(", ".join(parameter_sets.keys()))

    @staticmethod
    def print_refmsg():
        """Print message with licensing and reference information"""
        print(_references_msg)
