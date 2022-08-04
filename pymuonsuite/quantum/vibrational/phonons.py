"""
Author: Adam Laverack and Simone Sturniolo
"""


from collections import namedtuple

import numpy as np
import scipy.constants as cnst

from ase.dft.kpoints import monkhorst_pack
from ase.optimize import BFGS
from ase.phonons import Phonons
from ase.vibrations import Vibrations

ASEPhononData = namedtuple(
    "ASEPhononData", ["frequencies", "modes", "path", "structure"]
)


def ase_phonon_calc(
    struct,
    calc=None,
    kpoints=[1, 1, 1],
    ftol=0.01,
    force_clean=False,
    name="asephonon",
):
    """Calculate phonon modes of a molecule using ASE and a given calculator.
    The system will be geometry optimized before calculating the modes. A
    report of the phonon modes will be written to a file and arrays of the
    eigenvectors and eigenvalues returned.

    | Args:
    |   struct (ase.Atoms):     Atoms object with to calculate modes for.
    |   calc (ase.Calculator):  Calculator for energies and forces (if not
    |                           present, use the one from struct)
    |   kpoints (np.ndarray):   Kpoint grid for phonon calculation. If None, just
    |                           do a Vibration modes calculation (default is [1,1,1])
    |   ftol (float):           Tolerance for geometry optimisation (default
    |                           is 0.01 eV/Ang)
    |   force_clean (bool):     If True, force a deletion of all phonon files
    |                           and recalculate them
    | Returns:
    |   evals (float[k-points][modes]):          Eigenvalues of phonon modes
    |   evecs (float[k-points][modes][ions][3]): Eigenvectors of phonon modes
    |   struct (ase.Atoms):                      Optimised structure
    """

    N = len(struct)
    if calc is None:
        calc = struct.calc
    struct = struct.copy()
    calc.atoms = struct
    struct.calc = calc
    dyn = BFGS(struct, trajectory="geom_opt.traj")
    dyn.run(fmax=ftol)

    # Calculate phonon modes
    vib_pbc = kpoints is not None
    if vib_pbc:
        vib = Phonons(struct, calc, name=name)
    else:
        vib = Vibrations(struct, name=name)
    if force_clean:
        vib.clean()
    vib.run()
    if vib_pbc:
        vib.read(acoustic=True)
        path = monkhorst_pack(kpoints)
        evals, evecs = vib.band_structure(path, True)
    else:
        vib.read()
        path = np.zeros((1, 3))
        # One axis added since it's like the gamma point
        evals = np.real(vib.get_energies()[None])
        evecs = np.array([vib.get_mode(i) for i in range(3 * N)])[None]

    # eV to cm^-1
    evals *= ((cnst.electron_volt / cnst.h) / cnst.c) / 100.0
    # Normalise eigenvectors
    evecs /= np.linalg.norm(evecs, axis=(2, 3))[:, :, None, None]

    return ASEPhononData(evals, evecs, path, struct)


def get_apr(evecs, masses):
    """Compute Atomic Participation Ratios (APR) for all given phonon modes.

    | Args:
    |   evecs (Numpy float array, shape: (num_modes, num_ions, 3)):
    |                                   Eigenvectors of phonon modes of system
    |   masses (Numpy float array, shape: (num_ions)):
    |                                   Ionic masses
    |
    | Returns:
    |   APR (Numpy float array, shape: (num_modes, num_ions)):
    |                                   Matrix of Atomic Participation Ratios
    """

    evecs = np.array(evecs)
    masses = np.array(masses)
    Nk = evecs.shape[0]
    ek2 = np.sum(evecs**2, axis=2) / masses[None, :]
    return (ek2) / (Nk * np.sum(ek2**2, axis=1))[:, None] ** 0.5


def get_major_emodes(evecs, masses, i, n=3, ortho=False):
    """Find the n phonon modes with highest Atomic Participation Ratio (APR) for
    the atom at index i. Return orthogonalized and normalized modes if
    ortho is True.

    | Args:
    |   evecs (Numpy float array, shape: (num_modes, num_ions, 3)):
    |                                   Eigenvectors of phonon modes of system
    |   masses (Numpy float array, shape: (num_ions)):
    |                                   Ionic masses
    |   i (int): Index of atom in position array
    |   n (int): Number of eigenmodes to return (default is 3)
    |   ortho (bool): If true, orthogonalize and normalize major modes before
    |       returning.
    |
    | Returns:
    |   maj_evecs_i (int[3]): Indices of atom's phonon eigenvectors in evecs
    |   maj_evecs (float[3, 3]): Eigenvectors of atom's phonon modes
    """
    # First, find the atomic participation ratios for all atoms
    apr = get_apr(evecs, masses)
    evecs_order = np.argsort(apr[:, i])

    # How many?
    maj_evecs_i = evecs_order[-n:]
    maj_evecs = evecs[maj_evecs_i, i]

    if ortho:
        # Orthogonolize and normalize
        maj_evecs[0] = maj_evecs[0] / np.linalg.norm(maj_evecs[0])
        maj_evecs[1] = maj_evecs[1] - np.dot(maj_evecs[0], maj_evecs[1]) * maj_evecs[0]
        maj_evecs[1] = maj_evecs[1] / np.linalg.norm(maj_evecs[1])
        maj_evecs[2] = (
            maj_evecs[2]
            - np.dot(maj_evecs[0], maj_evecs[2]) * maj_evecs[0]
            - np.dot(maj_evecs[2], maj_evecs[1]) * maj_evecs[1]
        )
        maj_evecs[2] = maj_evecs[2] / np.linalg.norm(maj_evecs[2])

    return maj_evecs_i, maj_evecs.real
