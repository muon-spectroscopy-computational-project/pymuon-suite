"""
hfine.py

Computing hyperfine couplings for atomic structures with localised dipole
moments
"""

import numpy as np
import scipy.constants as cnst
from soprano.utils import minimum_supcell, supcell_gridgen
from soprano.properties.nmr.utils import _get_isotope_data


def compute_hfine_tensor(points, spins, cell=None, i=0, species='e'):
    """Compute the hyperfine tensor experienced at point of index i generated
    by a number of localised spins at points, for a given periodic unit cell
    and species.

    | Args:
    |   points (np.ndarray): coordinates of points at which the spins are l
    |                        localised
    |   spins (np.ndarray):  magnetic moments (in Bohr magnetons for electrons
    |                        or hbar*gamma units for nuclei)
    |   cell (np.ndarray):   unit cell (if None, considered non-periodic)
    |   i (int):             index of point at which to compute the tensor.
    |                        Local spin density will give rise to a Fermi 
    |                        contact term
    |   species (str or [str]): symbol or list of symbols identifying the
    |                           species generating the magnetic field.
    |                           Determines the magnetic moments

    | Returns:
    |   HT (np.ndarray):    hyperfine tensor at point i
    """

    pass
