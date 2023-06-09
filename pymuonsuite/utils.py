import shutil
import os
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
