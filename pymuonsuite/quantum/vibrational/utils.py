"""
utils.py

Utility functions for vibrational averages
"""

import numpy as np
from pymuonsuite.io.castep import parse_castep_masses
from pymuonsuite.constants import m_mu_amu


def apply_muon_mass(cell, mu_symbol=None, mu_index=-1):

    # Fetch species
    try:
        species = cell.get_array('castep_custom_species')
    except KeyError:
        species = np.array(cell.get_chemical_symbols())

    mu_indices = np.where(species == mu_symbol)[0]
    if len(mu_indices) > 1:
        raise MuonAverageError(
            'More than one muon found in the system')
    elif len(mu_indices) == 1:
        mu_index = mu_indices[0]

    if mu_symbol is not None:
        species = list(species)
        species[mu_index] = mu_symbol
        species = np.array(species)
        cell.set_array('castep_custom_species', species)

    # Set masses
    # Fetch masses
    try:
        masses = parse_castep_masses(cell)
    except AttributeError:
        # Just fall back on ASE standard masses if not available
        masses = cell.get_masses()
    masses[mu_index] = m_mu_amu
    cell.set_masses(masses)

    return cell, mu_index
