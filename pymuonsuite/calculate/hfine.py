"""hfine.py

Computing hyperfine couplings for atomic structures with localised dipole
moments
"""

import numpy as np
import scipy.constants as cnst
from soprano.utils import minimum_supcell, supcell_gridgen, minimum_periodic

try:
    from soprano.nmr.utils import _get_isotope_data
except ImportError:
    # Older versions
    from soprano.properties.nmr.utils import _get_isotope_data

from pymuonsuite import constants as mu_cnst

_bohrmag = cnst.physical_constants["Bohr magneton"][0]


def compute_hfine_tensor(
    points,
    spins,
    cell=None,
    self_i=0,
    species="e",
    cut_r=10,
    lorentz=True,
    fermi_mm=0,
):
    """Compute the hyperfine tensor experienced at point of index i generated
    by a number of localised spins at points, for a given periodic unit cell
    and species.

    | Args:
    |   points (np.ndarray): coordinates of points at which the spins are
    |                        localised
    |   spins (np.ndarray):  magnetic moments (as spin quantum number, e.g.
    |                        0.5 for an electron or 1H nucleus with spin up)
    |   cell (np.ndarray):   unit cell (if None, considered non-periodic)
    |   self_i (int):        index of point at which to compute the tensor.
    |                        Local spin density will give rise to a Fermi
    |                        contact term
    |   species (str or [str]): symbol or list of symbols identifying the
    |                           species generating the magnetic field.
    |                           Determines the magnetic moments
    |   cut_r (float):       cutoff radius for dipolar component calculation
    |   lorentz (bool):      if True, include a Lorentz term (average bulk
    |                        magnetization). Default is True
    |   fermi_mm (float):    Magnetic moment density at site i to use for
    |                        computation of the Fermi contact term. Units
    |                        are Bohr magnetons/Ang^3

    | Returns:
    |   HT (np.ndarray):    hyperfine tensor at point i
    """

    N = len(points)

    magmoms = np.array(spins).astype(float)
    species = np.array(species).astype("S2")
    if species.shape == ():
        species = np.repeat(species[None], N)
    for i, s in enumerate(species):
        if s == b"e":
            mm = 2 * _bohrmag
        elif s == b"mu":
            mm = mu_cnst.m_gamma * cnst.hbar
        else:
            mm = _get_isotope_data(s, "gamma")[0] * cnst.hbar
        magmoms[i] *= mm * abs(cnst.physical_constants["electron g factor"][0])

    # Do we need a supercell?
    r = np.array(points) - points[self_i]
    if cell is not None:
        scell = minimum_supcell(cut_r, latt_cart=cell)
        fxyz, xyz = supcell_gridgen(cell, scell)
        r = r[:, None, :] + xyz[None, :, :]
    else:
        r = r[:, None, :]

    rnorm = np.linalg.norm(r, axis=-1)
    # Expunge the ones that are outside of the sphere
    sphere = np.where(rnorm <= cut_r)
    r = r[sphere]
    rnorm = rnorm[sphere]
    magmoms = magmoms[sphere[0]]

    # Find the contact point
    self_i = np.argmin(rnorm)
    magmoms[self_i] = 0

    rnorm_inv = 1.0 / np.where(rnorm > 0, rnorm, np.inf)
    rdyad = r[:, None, :] * r[:, :, None]
    rdip = 3 * rdyad * rnorm_inv[:, None, None] ** 2 - np.eye(3)[None, :, :]

    HT = np.sum(magmoms[:, None, None] * rdip * rnorm_inv[:, None, None] ** 3, axis=0)
    HT *= cnst.mu_0 / (4 * np.pi) * 1e30

    # Add Lorentz term
    if cell is not None and lorentz:
        avgM = np.sum(magmoms) * 3.0 / (4.0 * np.pi * cut_r**3)
        HT += np.eye(3) * avgM * cnst.mu_0 / 3.0 * 1e30
    # Add contact term
    if fermi_mm:
        fermi_mm *= _bohrmag * abs(cnst.physical_constants["electron g factor"][0])
        HT += np.eye(3) * fermi_mm * 2.0 / 3.0 * cnst.mu_0 * 1e30

    return HT


def compute_hfine_mullpop(
    atoms,
    populations,
    self_i=0,
    cut_r=10,
    lorentz=True,
    fermi=True,
    fermi_neigh=False,
):
    """Compute a hyperfine tensor for a given atomic system from the Mulliken
    electronic populations, with additional empyrical corrections.

    | Args:
    |   atoms (ase.Atoms):   Atoms object to work with
    |   populations (np.ndarray):  electronic orbital-resolved Mulliken
    |                              populations, as returned for example by
    |                              parse_spinpol_dftb. Spins here ought to
    |                              be in Bohr magnetons (so for example
    |                              one electron up would pass as 0.5)
    |   self_i (int):        index of point at which to compute the tensor.
    |                        Local spin density will give rise to a Fermi
    |                        contact term
    |   species (str or [str]): symbol or list of symbols identifying the
    |                           species generating the magnetic field.
    |                           Determines the magnetic moments
    |   cut_r (float):       cutoff radius for dipolar component calculation
    |   lorentz (bool):      if True, include a Lorentz term (average bulk
    |                        magnetization). Default is True
    |   fermi (bool):        if True, include a Fermi contact term
    |                        (magnetization at site i). Default is True
    |   fermi_neigh (bool):  if True, include an empyrical neighbour
    |                        correction for the Fermi contact term.
    |                        Default is False

    | Returns:
    |   HT (np.ndarray):    hyperfine tensor at point i
    """

    pbc = atoms.get_pbc()
    # Only works if either all, or none
    if pbc.any() and not pbc.all():
        raise ValueError("Partially periodic systems not implemented")
    pbc = pbc.all()
    if pbc:
        cell = atoms.get_cell()
    else:
        cell = None
    pos = atoms.get_positions()

    # First correction: compile the total spins
    totspins = np.array([sp["spin"] for sp in populations])
    magmoms = totspins

    # Fermi contact term density
    fermi_mm = 0
    if fermi:
        a0 = cnst.physical_constants["Bohr radius"][0]
        Z = atoms.get_atomic_numbers()[self_i]
        for (n, l, m), p in populations[self_i]["spin_orbital"].items():
            if l > 0:
                continue
            fermi_mm += 2 / np.pi * (Z / (n * a0 * 1e10) ** 3.0) * p
        # If required, add the neighbour effect
        if fermi_neigh:
            dr = pos - pos[self_i]
            if pbc:
                dr, _ = minimum_periodic(dr, cell)
            drnorm = np.linalg.norm(dr, axis=-1)

            # This is totally empirical! Works with C6H6Mu for now
            expcorr = np.exp(-drnorm * 1e-10 / (1.55 * a0)) * totspins
            expcorr[self_i] = 0.0

            fermi_mm += np.sum(expcorr)

    return compute_hfine_tensor(
        pos,
        magmoms,
        cell,
        self_i=self_i,
        cut_r=cut_r,
        lorentz=lorentz,
        fermi_mm=fermi_mm,
    )
