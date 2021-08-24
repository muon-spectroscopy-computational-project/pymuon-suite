"""
constants.py

Useful constants
"""

import scipy.constants as cnst

m_mu = cnst.physical_constants["muon mass"][0]
m_mu_amu = cnst.physical_constants["muon mass in u"][0]
# Made positive because we need positive muons
m_gamma = abs(cnst.physical_constants["muon mag. mom."][0] * 2 / cnst.hbar)
