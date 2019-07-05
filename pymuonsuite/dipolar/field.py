"""
field.py

DipolarField class, computing dipolar field distributions at a muon position
"""

# Python 2-to-3 compatibility code
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import scipy.constants as cnst
from soprano.utils import minimum_supcell, supcell_gridgen
from soprano.calculate.powder import ZCW, SHREWD, TriAvg
from soprano.properties.nmr.utils import _get_isotope_data, _dip_constant
from pymuonsuite.constants import m_gamma


class DipolarField(object):

    def __init__(self, atoms, mu_pos, isotopes={}, isotope_list=None, cutoff=10):

        # Get positions, cell, and species, only things we care about
        self.cell = np.array(atoms.get_cell())
        pos = atoms.get_positions()
        el = np.array(atoms.get_chemical_symbols())

        self.mu_pos = np.array(mu_pos)

        scell = minimum_supcell(cutoff, self.cell)
        grid_f, grid = supcell_gridgen(self.cell, scell)

        self.grid_f = grid_f


        r = (pos[:, None, :]+grid[None, :, :]-self.mu_pos[None, None, :]).reshape((-1,3))
        rnorm = np.linalg.norm(r, axis=1)
        sphere = np.where(rnorm <= cutoff)[0]
        sphere = sphere[np.argsort(rnorm[sphere])[::-1]] # Sort by length
        r = r[sphere]
        rnorm = rnorm[sphere]

        self._r = r
        self._rn = rnorm
        self._ri = sphere
        self._dT = 3*r[:,:,None]*r[:,None,:]/rnorm[:,None,None]**2-np.eye(3)[None,:,:]

        # Get gammas
        self.gammas = _get_isotope_data(el, 'gamma', isotopes, isotope_list)

    def get_single_field(self, moments, ext_field_dir=[0,0,1], moment_type='e'):

        gn = self.grid_f.shape[0]
        a_i = self._ri//gn
        ijk = self.grid_f[self._ri%gn]

        s = np.array(moments)[a_i]
        if moment_type == 'n':
            gammas = self.gammas[a_i]
        elif moment_type == 'e':
            gammas = cnst.physical_constants['electron gyromag. ratio'][0]

        D = _dip_constant(self._rn*1e-10, m_gamma, gammas)

        n = np.array(ext_field_dir)
        n /= np.linalg.norm(n)

        DT = np.sum((D*s)[:,None,None]*self._dT, axis=0)
        return np.dot(n, np.dot(DT, n))