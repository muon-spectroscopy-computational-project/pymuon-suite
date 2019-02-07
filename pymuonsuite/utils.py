# Python 2-to-3 compatibility code
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import shutil
import numpy as np
from ase import Atoms
from soprano.utils import minimum_periodic

def find_ipso_hydrogen(a_i, cell, symbol):
    """Find closest hydrogen to atom at index a_i in cell

    | Args:
    |   a_i (int): Index of atom in cell position array
    |   cell (Atoms object): ASE atoms object for molecule
    |   symbol (str): Symbol used to represent relevant atom in cell
    |
    | Returns:
    |   ipso_i (int): Index of closest hydrogen in cell position array
    """
    if cell.has('castep_custom_species'):
        chems = cell.get_array('castep_custom_species')
    else:
        chems = np.array(cell.get_chemical_symbols())
    pos = cell.get_positions()
    iH = np.where(['H' in c and c != symbol for c in chems])[0]
    posH = pos[iH]
    distH = np.linalg.norm(
        minimum_periodic(posH - pos[a_i], cell.get_cell())[0], axis=-1)
    #Which one is the closest?
    ipso_i = iH[np.argmin(distH)]

    return ipso_i

def list_to_string(arr):
    """Create a str from a list an array of list of numbers

    | Args:
    |   arr (list): a list of numbers to convert to a space
    |       seperated string
    |
    | Returns:
    |    string (str): a space seperated string of numbers
    """
    return ' '.join(map(str, arr))


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
        raise ValueError('Invalid argument passed do make_3x3')


def safe_create_folder(path):
    """Create a folder at path with safety checks for overwriting.

    | Args:
    |   path (str): path at which to create the new folder
    |
    | Returns:
    |   success (bool): True if the operation was successful

    """

    while os.path.isdir(path):
        ans = raw_input(('Folder {} exists, overwrite (y/N)? '
                         ).format(path))
        if ans == 'y':
            shutil.rmtree(path)
        else:
            return False
    try:
        os.mkdir(path)
    except OSError:
        return False

    return True


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
    f01 = f1-f0
    if np.isclose(np.linalg.norm(f01), 0):
        raise ValueError('Points f0 and f1 are too close')
    if not np.isclose(np.dot(hkl, f01), 0):
        raise ValueError('Points f0 and f1 do not belong to the same plane')

    # Now move to direct space
    n = np.dot(hkl, np.linalg.inv(cell))
    p0 = np.dot(cell.T, f0)
    p1 = np.dot(cell.T, f1)

    n /= np.linalg.norm(n)

    # Find the scanning directions
    p01 = p1-p0

    plx = np.zeros(3)
    plx[np.where(p01 != 0)[0][0]] = 1
    ply = p01 - np.dot(p01, plx)*plx
    ply /= np.linalg.norm(ply)
    ply *= np.sign(ply[np.where(ply != 0)[0][0]])

    # Now to actually create the scanning grid
    plgrid = np.array(np.meshgrid(*[np.linspace(0, 1, N)]*2, indexing='ij'))

    xyzgrid = plgrid[0, None]*plx[:, None, None]*np.dot(p01, plx)
    xyzgrid += plgrid[1, None]*ply[:, None, None]*np.dot(p01, ply)
    xyzgrid += p0[:, None, None]

    return xyzgrid


class BackupFile():
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
