import os
import shutil

from ase import Atoms
from ase.build import make_supercell as make_supercell_ase

import numpy as np

from soprano.utils import safe_input


def list_to_string(arr):
    """Create a str from a list an array of list of numbers

    | Args:
    |   arr (list): a list of numbers to convert to a space
    |       seperated string
    |
    | Returns:
    |    string (str): a space seperated string of numbers
    """
    return " ".join(map(str, arr))


def make_3x3(a):
    """Reshape a into a 3x3 matrix. If it's a single number, multiply it by
    the identity matrix; if it's three numbers, make them into a diagonal
    matrix; and if it's nine, reshape them into a 3x3 matrix. Anything else
    gives rise to an exception.

    | Args:
    |   a (int, float or list): either a single number, a list of
    |                           numbers of size 1, 3, or 9, or a 2D list of
    |                           size 9
    |
    | Returns:
    |   matrix (np.array): a 2D numpy matrix of shape (3,3)
    """

    # parameter is some shape of list
    a = np.array(a, dtype=int)
    if a.shape == (1,) or a.shape == ():
        return np.eye(3) * a
    elif a.shape == (3,):
        return np.diag(a)
    elif a.shape == (9,) or a.shape == (3, 3):
        return a.reshape((3, 3))
    else:
        # All failed
        raise ValueError("Invalid argument passed do make_3x3")


def make_supercell(structure: Atoms, supercell: int) -> Atoms:
    """Expand the structure into the defined supercell. If supercell is a single
    number, it is expanded along each axis by the specified factor. Lists of 3
    or 9 elements are also supported, in which case the cell will be expanded by
    the amount corresponding to each direction.

    | Args:
    |   structure (Atoms): the structure to be expanded
    |   supercell (int, float or list): either a single number, a list of
    |                                   numbers of size 1, 3, or 9, or a 2D
    |                                   list of size 9
    |
    | Returns:
    |   Atoms: an ASE Atoms object with expanded size
    """
    supercell_matrix = make_3x3(supercell)
    # ASE's make_supercell is weird, avoid if not necessary...
    supercell_matrix_diag = np.diag(supercell_matrix).astype(int)
    if np.all(np.diag(supercell_matrix_diag) == supercell_matrix):
        return structure.repeat(supercell_matrix_diag)
    else:
        return make_supercell_ase(structure, supercell_matrix)


def make_muonated_supercell(structure: Atoms, supercell: int, mu_symbol: str) -> Atoms:
    """Expand a structure containing a muon into the defined supercell. The
    muon is not duplicated in the expansion (i.e. the only muon in the output
    will be at its original position).

    If supercell is a single number, it is expanded along each axis by the
    specified factor. Lists of 3 or 9 elements are also supported, in which
    case the cell will be expanded by the amount corresponding to each
    direction.

    | Args:
    |   structure (Atoms): the structure to be expanded, with the muon being
    |                      the final atom in the structure
    |   supercell (int, float or list): either a single number, a list of
    |                                   numbers of size 1, 3, or 9, or a 2D
    |                                   list of size 9
    |   mu_symbol (str): castep_custom_species to use for the muon
    |
    | Returns:
    |   Atoms: an ASE Atoms object with expanded size"""
    structure = structure.copy()  # Avoid in place modifications
    muon = structure.pop()
    supercell_structure = make_supercell(structure, supercell)
    chemical_symbols = supercell_structure.get_chemical_symbols()
    castep_custom_species = np.array(chemical_symbols + [mu_symbol])
    supercell_structure += muon
    supercell_structure.set_array("castep_custom_species", castep_custom_species)
    return supercell_structure


def safe_create_folder(folder_name):
    """Create a folder at path with safety checks for overwriting.

    | Args:
    |   path (str): path at which to create the new folder
    |
    | Returns:
    |   success (bool): True if the operation was successful

    """
    while os.path.isdir(folder_name):
        ans = safe_input(("Folder {} exists, overwrite (y/N)? ").format(folder_name))
        if ans == "y":
            shutil.rmtree(folder_name)
        else:
            folder_name = safe_input("Please input new folder name:\n")
    try:
        os.mkdir(folder_name)
    except OSError:
        pass  # It's fine, it already exists
    return folder_name


def make_process_slices(N, M):
    if M is not None:
        sl0 = np.arange(0, N, M)
        sl1 = sl0 + M
        slices = map(lambda s: slice(*s), zip(sl0, sl1))
    else:
        slices = [slice(0, N)]

    return slices


def create_plane_grid(hkl, cell, f0, f1, N=20):
    # Create a grid of points along a crystal plane

    hkl = np.array(hkl).astype(int)
    f0 = np.array(f0)
    f1 = np.array(f1)

    # First: verify that the given points (given in fractional coordinates)
    # DO belong to the same plane
    f01 = f1 - f0
    if np.isclose(np.linalg.norm(f01), 0):
        raise ValueError("Points f0 and f1 are too close")
    if not np.isclose(np.dot(hkl, f01), 0):
        raise ValueError("Points f0 and f1 do not belong to the same plane")

    # Now move to direct space
    n = np.dot(hkl, np.linalg.inv(cell))
    p0 = np.dot(cell.T, f0)
    p1 = np.dot(cell.T, f1)

    n /= np.linalg.norm(n)

    # Find the scanning directions
    p01 = p1 - p0

    plx = np.zeros(3)
    plx[np.where(p01 != 0)[0][0]] = 1
    ply = p01 - np.dot(p01, plx) * plx
    ply /= np.linalg.norm(ply)
    ply *= np.sign(ply[np.where(ply != 0)[0][0]])

    # Now to actually create the scanning grid
    plgrid = np.array(np.meshgrid(*[np.linspace(0, 1, N)] * 2, indexing="ij"))

    xyzgrid = plgrid[0, None] * plx[:, None, None] * np.dot(p01, plx)
    xyzgrid += plgrid[1, None] * ply[:, None, None] * np.dot(p01, ply)
    xyzgrid += p0[:, None, None]

    return xyzgrid


class BackupFile:
    """Backup a file before performing an operation

    A class to make a copy of a file before performing some
    potentially unsafe operation on it. In either succes or failure
    the copy of the original file it restored.
    """

    def __init__(self, file_name, backup_file):
        """Create a temporary backup file while executing some
        potentially unsafe operation.

        Args:
          file_name (str): path of the file to backup.
          backup_file (str): path to backup the file to.
        """
        self._file_name = file_name
        self._backup_file = backup_file

    def __enter__(self):
        """Copy the file to the backup location"""
        shutil.copyfile(self._file_name, self._backup_file)
        return self

    def __exit__(self, type, value, traceback):
        """Replace and overwrite the file to the original location"""
        shutil.move(self._backup_file, self._file_name)
