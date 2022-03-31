"""charged.py

ChargeDistribution class for Unperturbed Electrostatic Potential
"""


import os

import numpy as np
from ase import io
from ase.calculators.singlepoint import SinglePointCalculator
from parsefmt.fmtreader import FMTReader
from pymuonsuite.io.castep import parse_castep_ppots
from pymuonsuite.utils import make_process_slices
from scipy import constants as cnst
from soprano.utils import silence_stdio

# Coulomb constant
_cK = 1.0 / (4.0 * np.pi * cnst.epsilon_0)
# Convert dipolar couplings to Tesla
_dipT = (
    cnst.mu_0
    / (4 * np.pi)
    * cnst.physical_constants["Bohr magneton"][0]
    * 1e30
    * abs(cnst.physical_constants["electron g factor"][0])
)
# Convert Fermi contact term to Tesla
_fermiT = (
    2.0
    / 3.0
    * cnst.mu_0
    * cnst.physical_constants["Bohr magneton"][0]
    * 1e30
    * abs(cnst.physical_constants["electron g factor"][0])
)


class ChargeDistribution(object):
    """ChargeDistribution

    An object storing a distribution of electronic and ionic charges in a
    unit cell, and using it to compute the electrostatic potential. Currently
    needs a CASTEP .den_fmt file as input.
    Ions are described approximately as Gaussian distributions of positive
    charge Z-e_core, where e_core is the number of core electrons included
    in the pseudopotential.
    """

    def __init__(self, seedname, gw_fac=3, path=""):
        """Initialise a ChargeDistrbution object.

        Initialise a ChargeDistribution object with CASTEP output files.

        Arguments:
            seedname {str} -- The seedname of the CASTEP output files
                              (.den_fmt and .castep) used to load the data.

        Keyword Arguments:
            gw_fac {number} -- Gaussian width factor. The Gaussian width used
                               for each ion will be the radius of its
                               pseudopotential divided by this factor.
                               (default: {3})
            path {str} -- Path in which the CASTEP output files can be found.
                          (default: {''})

        Raises:
            RuntimeError -- CASTEP pseudopotentials were not found
        """

        # CASTEP SPECIFIC CODE - reading files

        # Load the electronic density
        seedpath = os.path.join(path, seedname)

        self._elec_den = FMTReader(seedpath + ".den_fmt")
        with silence_stdio():
            self._struct = io.read(seedpath + ".castep")

        ppots = parse_castep_ppots(seedpath + ".castep")

        # Override by also grabbing any pseudopotentials found in the .cell
        # file

        cppot = None
        try:
            with silence_stdio():
                cppot = io.read(seedpath + ".cell").calc.cell.species_pot.value
        except IOError:
            pass  # If not available, ignore this
        if cppot is not None:
            ppf = [lpp.split() for lpp in cppot.split("\n") if lpp]
            for el, pppath in ppf:
                f = os.path.join(path, pppath)
                try:
                    ppots.update(parse_castep_ppots(f))
                except IOError:
                    # File not found
                    print("WARNING: pseudopotential file " "{0} not found".format(f))

        # END OF CASTEP SPECIFIC CODE

        # unit cell and FFT grid
        lattice = np.array(self._elec_den.real_lattice)  # Ang
        grid = np.array(self._elec_den.grid)

        # dx = [np.linalg.norm(lattice[i]) / grid[i] for i in range(3)]
        inv_latt = np.linalg.inv(lattice.T) * 2 * np.pi  # Ang^-1

        fft_grid = np.array(
            np.meshgrid(
                *[np.fft.fftfreq(grid[i]) * grid[i] for i in range(3)], indexing="ij"
            )
        )
        # Uses the g-vector convention in formulas used
        # this should match CASTEP's grid in reciprocal space
        self._g_grid = np.tensordot(inv_latt, fft_grid, axes=(0, 0))  # Ang^-1

        # Information for the elements, and guarantee zero net charge
        elems = self._struct.get_chemical_symbols()
        pos = self._struct.get_positions()
        try:
            # CASTEP SPECIFIC CODE (maybe) - this way of getting charges &
            # gaussian widths from pseudopotentials
            self._q = np.array([ppots[el][0] for el in elems])  # e
            self._gw = np.array([ppots[el][1] / gw_fac for el in elems])  # Ang
            # END OF CASTEP SPECIFIC CODE
        except KeyError:
            raise RuntimeError(
                """Some or all CASTEP pseudopotentials were not
found. UEP calculation can not go on. Please notice that at the moment only
ultrasoft pseudopotentials are supported, and if not generated automatically,
they must be possible to retrieve using the paths in the SPECIES_POT block of
the .cell file."""
            )

        # Here we find the Fourier components of the potential due to
        # the valence electrons
        self._rho = self._elec_den.data[:, :, :, 0]
        if not np.isclose(np.average(self._rho), sum(self._q), 1e-4):
            raise RuntimeError("Cell is not neutral")
        # Put the minus sign for electrons

        # CASTEP SPECIFIC CODE - unit conversion. Needs standardising.
        self._rho *= -sum(self._q) / np.sum(
            self._rho
        )  # Normalise charge. Now it's e/grid points
        # END OF CASTEP SPECIFIC CODE

        self._rhoe_G = np.fft.fftn(self._rho)
        Gnorm = np.linalg.norm(self._g_grid, axis=0)
        # remove 0-values as we will use 1/G_norm in calculations
        # (0-values are special cases and we set the results manually elsewhere)
        Gnorm_fixed = np.where(Gnorm > 0, Gnorm, np.inf)

        cell = np.array(self._elec_den.real_lattice)
        vol = abs(np.dot(np.cross(cell[:, 0], cell[:, 1]), cell[:, 2]))  # Ang^3
        self._vol = vol

        # Core formula: reciprocal space potential contribution FROM ELECTRONS
        self._Ve_G = 4 * np.pi / Gnorm_fixed**2 * (self._rhoe_G / vol)  # e/Ang

        # Now on to doing the same for ionic components
        self._rhoi_G = self._g_grid[0] * 0.0j
        for i, p in enumerate(pos):
            self._rhoi_G += self._q[i] * np.exp(
                -1.0j  # phase term corresponding to translation in direct space
                * np.sum(self._g_grid[:, :, :, :] * p[:, None, None, None], axis=0)
                - 0.5 * (self._gw[i] * Gnorm) ** 2
            )

        # Reciprocal space potential contributions FROM IONS
        # (approximated as Gaussian charges)
        self._Vi_G = 4 * np.pi / Gnorm_fixed**2 * (self._rhoi_G / vol)

        # Is there any data on spin polarization?
        self._spinpol = False
        if self._elec_den.data.shape[-1] >= 2:
            self._spinpol = True
            self._spin = self._elec_den.data[:, :, :, 1]
            self._spin_G = np.fft.fftn(self._spin)

            # Dipolar tensor FFT
            dyad_G = self._g_grid[:, None] * self._g_grid[None, :]
            dyad_G /= Gnorm_fixed**2
            self._dip_G = (
                4.0
                / 3.0
                * np.pi
                * (3 * dyad_G - np.eye(3)[:, :, None, None, None] + 0j)
            )
            self._dip_G[:, :, 0, 0, 0] = 0
            self._dip_G *= self._spin_G / (self._vol * np.prod(self._spin.shape))
            # Convert to Tesla.
            self._dip_G *= _dipT

        # Now, Thomas-Fermi energy
        tfint = np.sum(abs(self._rho / self._vol) ** (5 / 3) * self._vol)
        C = (
            0.3
            * cnst.hbar**2
            / cnst.m_e
            * (3 * np.pi**2) ** (2 / 3)
            / cnst.e
            * 1e20
        )
        self._thomasFermiE = C * tfint

    @property
    def atoms(self):
        a = self._struct.copy()
        E, F = self._struct.get_potential_energy(), self._struct.get_forces()
        a.calc = SinglePointCalculator(atoms=a, energy=E, forces=F)
        return a

    @property
    def cell(self):
        return self._struct.get_cell()

    @property
    def volume(self):
        return self._vol

    @property
    def chemical_symbols(self):
        return self._struct.get_chemical_symbols()

    @property
    def positions(self):
        return self._struct.get_positions()

    @property
    def scaled_positions(self):
        return self._struct.get_scaled_positions()

    @property
    def has_spin(self):
        return self._spinpol

    @property
    def thomasFermiE(self):
        return self._thomasFermiE

    def rho(self, p, max_process_p=20):
        """Charge density

        Compute charge density at a point or list of points, total and
        split by electronic and ionic contributions.

        Arguments:
            p {np.ndarray} -- List of points to compute charge density at.

        Keyword Arguments:
            max_process_p {number} -- Max number of points processed at once.
                                      Lower to trade off speed for memory
                                      (default: {20})

        Returns:
            np.ndarray -- Total charge density
            np.ndarray -- Electronic charge density
            np.ndarray -- Ionic charge density
        """

        # Return charge density at a point or list of points
        p = np.array(p)
        if len(p.shape) == 1:
            p = p[None, :]  # Make it into a list of points

        # The point list is sliced for convenience, to avoid taking too much
        # memory
        N = p.shape[0]
        rhoe = np.zeros(N)
        rhoi = np.zeros(N)

        slices = make_process_slices(N, max_process_p)

        for s in slices:
            # Fourier transform kernel
            ftk = np.exp(1.0j * np.tensordot(self._g_grid, p[s].T, axes=(0, 0)))
            rhoe[s] = np.real(np.sum(self._rhoe_G[:, :, :, None] * ftk, axis=(0, 1, 2)))
            rhoi[s] = np.real(np.sum(self._rhoi_G[:, :, :, None] * ftk, axis=(0, 1, 2)))

        # Convert units to e/Ang^3
        rhoe /= self._vol
        rhoi /= self._vol
        rho = rhoe + rhoi

        return rho, rhoe, rhoi

    def V(self, p, max_process_p=20):
        """Potential

        Compute electrostatic potential at a point or list of points,
        total and split by electronic and ionic contributions.

        Arguments:
            p {np.ndarray} -- List of points to compute potential at.

        Keyword Arguments:
            max_process_p {number} -- Max number of points processed at once.
                                      Lower to trade off speed for memory
                                      (default: {20})

        Returns:
            np.ndarray -- Total potential
            np.ndarray -- Electronic potential
            np.ndarray -- Ionic potential
        """

        # Return potential at a point or list of points
        p = np.array(p)
        if len(p.shape) == 1:
            p = p[None, :]  # Make it into a list of points

        # The point list is sliced for convenience, to avoid taking too much
        # memory
        N = p.shape[0]
        Ve = np.zeros(N)
        Vi = np.zeros(N)

        slices = make_process_slices(N, max_process_p)

        for s in slices:
            # Fourier transform kernel
            ftk = np.exp(1.0j * np.tensordot(self._g_grid, p[s].T, axes=(0, 0)))
            # Compute the electronic potential
            Ve[s] = np.real(np.sum(self._Ve_G[:, :, :, None] * ftk, axis=(0, 1, 2)))
            # Now add the ionic one
            Vi[s] = np.real(np.sum(self._Vi_G[:, :, :, None] * ftk, axis=(0, 1, 2)))

        Ve *= _cK * cnst.e * 1e10  # Moving to SI units
        Vi *= _cK * cnst.e * 1e10

        V = Ve + Vi

        return V, Ve, Vi

    def dV(self, p, max_process_p=20):
        """Potential gradient

        Compute electrostatic potential gradient at a point or list of
        points, total and split by electronic and ionic contributions.

        Arguments:
            p {np.ndarray} -- List of points to compute potential gradient at.

        Keyword Arguments:
            max_process_p {number} -- Max number of points processed at once.
                                      Lower to trade off speed for memory
                                      (default: {20})

        Returns:
            np.ndarray -- Total potential gradient
            np.ndarray -- Electronic potential gradient
            np.ndarray -- Ionic potential gradient
        """

        # Return potential gradient at a point or list of points
        p = np.array(p)
        if len(p.shape) == 1:
            p = p[None, :]  # Make it into a list of points

        # The point list is sliced for convenience, to avoid taking too much
        # memory
        N = p.shape[0]
        dVe = np.zeros((N, 3))
        dVi = np.zeros((N, 3))

        slices = make_process_slices(N, max_process_p)

        for s in slices:
            # Fourier transform kernel
            ftk = np.exp(1.0j * np.tensordot(self._g_grid, p[s].T, axes=(0, 0)))
            dftk = 1.0j * self._g_grid[:, :, :, :, None] * ftk[None, :, :, :, :]
            # Compute the electronic potential
            dVe[s] = np.real(
                np.sum(self._Ve_G[None, :, :, :, None] * dftk, axis=(1, 2, 3))
            ).T
            # Now add the ionic one
            dVi[s] = np.real(
                np.sum(self._Vi_G[None, :, :, :, None] * dftk, axis=(1, 2, 3))
            ).T

        dVe *= _cK * cnst.e * 1e20  # Moving to SI units
        dVi *= _cK * cnst.e * 1e20

        dV = dVe + dVi

        return dV, dVe, dVi

    def d2V(self, p, max_process_p=20):
        """Potential Hessian

        Compute electrostatic potential Hessian at a point or list of points,
        total and split by electronic and ionic contributions.

        Arguments:
            p {np.ndarray} -- List of points to compute potential Hessian at.

        Keyword Arguments:
            max_process_p {number} -- Max number of points processed at once.
                                      Lower to trade off speed for memory
                                      (default: {20})

        Returns:
            np.ndarray -- Total potential Hessian
            np.ndarray -- Electronic potential Hessian
            np.ndarray -- Ionic potential Hessian
        """

        # Return potential Hessian  at a point or a list of points

        p = np.array(p)
        if len(p.shape) == 1:
            p = p[None, :]  # Make it into a list of points

        # The point list is sliced for convenience, to avoid taking too much
        # memory
        N = p.shape[0]
        d2Ve = np.zeros((N, 3, 3))
        d2Vi = np.zeros((N, 3, 3))

        slices = make_process_slices(N, max_process_p)
        g2_mat = self._g_grid[:, None, :, :, :] * self._g_grid[None, :, :, :, :]

        for s in slices:
            # Fourier transform kernel
            ftk = np.exp(1.0j * np.tensordot(self._g_grid, p[s].T, axes=(0, 0)))
            d2ftk = -g2_mat[:, :, :, :, :, None] * ftk[None, None, :, :, :, :]
            # Compute the electronic potential
            d2Ve[s] = np.real(
                np.sum(
                    self._Ve_G[None, None, :, :, :, None] * d2ftk,
                    axis=(2, 3, 4),
                )
            ).T
            # Now add the ionic one
            d2Vi[s] = np.real(
                np.sum(
                    self._Vi_G[None, None, :, :, :, None] * d2ftk,
                    axis=(2, 3, 4),
                )
            ).T

        d2Ve *= _cK * cnst.e * 1e30  # Moving to SI units
        d2Vi *= _cK * cnst.e * 1e30

        d2V = d2Ve + d2Vi

        return d2V, d2Ve, d2Vi

    def Hfine(self, p, contact=False, max_process_p=20):
        """Hyperfine tensor

        Compute hyperfine tensor at a point or list of points. Only possible
        for electronic densities including spin polarisation.

        Arguments:
            p {np.ndarray} -- List of points to compute hyperfine tensor at.

        Keyword Arguments:
            contact {bool} -- If True, include Fermi contact term
                              (default: {False})
            max_process_p {number} -- Max number of points processed at once.
                                      Lower to trade off speed for memory
                                      (default: {20})

        Returns:
            np.ndarray -- Total hyperfine tensor
            np.ndarray -- Electronic hyperfine tensor
            np.ndarray -- Ionic hyperfine tensor

        Raises:
            RuntimeError -- If the electronic density is not spin polarised.
        """

        if not self.has_spin():
            raise RuntimeError(
                "Can not compute hyperfine tensor without"
                " spin polarised electronic density"
            )

        # Return hyperfine tensors at a point or a list of points
        p = np.array(p)
        if len(p.shape) == 1:
            p = p[None, :]  # Make it into a list of points

        # The point list is sliced for convenience, to avoid taking too much
        # memory
        N = p.shape[0]
        HT = np.zeros((N, 3, 3))

        slices = make_process_slices(N, max_process_p)

        for s in slices:
            # Fourier transform kernel
            ftk = np.exp(1.0j * np.tensordot(self._g_grid, p[s].T, axes=(0, 0)))
            # Compute the electronic potential
            HT[s] = np.real(
                np.sum(
                    self._dip_G[:, :, :, :, :, None] * ftk[None, None],
                    axis=(2, 3, 4),
                )
            ).T
            # And Fermi contact term
            if contact:
                fermi = np.real(
                    np.sum(self._spin_G[:, :, :, None] * ftk, axis=(0, 1, 2))
                )
                fermi *= _fermiT / (self._vol * np.prod(self._spin.shape))
                HT[s] += np.eye(3)[None, :, :] * fermi[:, None, None]

        return HT
