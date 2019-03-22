"""
charged.py

ChargeDistribution class for Perturbed Electrostatic Potential
"""

# Python 2-to-3 compatibility code
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import numpy as np
from scipy import constants as cnst
from soprano.utils import (max_distance_in_cell, minimum_supcell,
                           supcell_gridgen)

from pymuonsuite.utils import make_process_slices
from pymuonsuite.calculate.uep import ChargeDistribution


def _fk(x, k=0):
    if k <= 0:
        return x
    else:
        fkm = _fk(x, k-1)
        return 0.5*fkm*(3-fkm**2)


class PEPChargeDistribution(ChargeDistribution):

    def __init__(self, seedname, gw_fac=3, path=''):

        super(PEPChargeDistribution, self).__init__(seedname, gw_fac, path)

        # Partition the charges using the Becke method
        xyz = self._elec_den.xyz
        maxR = max_distance_in_cell(self.cell)
        scell = minimum_supcell(maxR, self.cell)
        scgfrac, scgxyz = supcell_gridgen(self.cell, scell)

        # For each point, find their minimum periodic distance from each ion
        pos = self.positions.T
        scgion = pos[:, :, None]+(scgxyz.T)[:, None, :]

        rA = np.zeros((xyz.shape[1], xyz.shape[2], xyz.shape[3],
                       len(self.positions)))

        for i in range(xyz.shape[1]):
            for j in range(xyz.shape[2]):
                rA[i, j, :, :] = np.amin(np.linalg.norm(xyz[:, i, j, :, None,
                                                            None]
                                                        - scgion[:, None, :, :],
                                                        axis=0), axis=-1)

        rAB = np.linalg.norm(pos[:, :, None]-pos[:, None, :], axis=0)
        rAB += np.diag([np.inf]*len(rAB))
        muAB = ((rA[:, :, :, :, None]-rA[:, :, :, None, :]) /
                rAB[None, None, None, :, :])

        sk = 0.5*(1.0-_fk(muAB, 3))
        # Set the diagonal to 1 so that it doesn't affect the products
        ii = range(len(self.positions))
        sk[:,:,:,ii, ii] = 1

        # Now weight functions
        wA = np.prod(sk, axis=-1)
        self._wA = wA/np.sum(wA, axis=-1)[:, :, :, None]

        # Finally, partition the density
        self._rhopart = self._rho[:, :, :, None]*self._wA
        self._rhopart_G = np.fft.fftn(self._rhopart, axes=(0, 1, 2))

        Gnorm = np.linalg.norm(self._g_grid, axis=0)
        Gnorm_fixed = np.where(Gnorm > 0, Gnorm, np.inf)

        vol = self.volume*np.prod(self._elec_den.grid)

        self._Vpart_G = 4*np.pi/Gnorm_fixed[:, :, :, None]**2*(self._rhopart_G /
                                                               vol)

    def rhopart(self, p, max_process_p=20):
        # Return partitioned charge density at a point or list of points
        p = np.array(p)
        if len(p.shape) == 1:
            p = p[None, :]   # Make it into a list of points

        # The point list is sliced for convenience, to avoid taking too much
        # memory
        N = p.shape[0]
        I = len(self.positions)
        rhoe = np.zeros((I, N))
        rhoi = np.zeros((I, N))

        slices = make_process_slices(N, max_process_p)

        for s in slices:
            # Fourier transform kernel
            ftk = np.exp(1.0j*np.tensordot(self._g_grid, p[s].T, axes=(0, 0)))
            rhoe[:, s] = np.real(np.sum(self._rhopart_G[:, :, :, :, None] *
                                        ftk[:, :, :, None],
                                        axis=(0, 1, 2)))
            rhoi[:, s] = np.real(np.sum(self._rhoi_G[:, :, :, :, None] *
                                        ftk[:, :, :, None],
                                        axis=(0, 1, 2)))
        # Convert units to e/Ang^3
        rhoe /= self._vol
        rhoi /= self._vol
        rho = rhoe+rhoi

        return rho, rhoe, rhoi
