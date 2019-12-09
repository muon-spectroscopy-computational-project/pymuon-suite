# Python 2-to-3 compatibility code
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import pickle
import numpy as np
import argparse as ap
import datetime

from ase import io
from scipy.optimize import minimize
from scipy import constants as cnst


from pymuonsuite.schemas import UEPOptSchema, load_input_file
from pymuonsuite.calculate.uep.charged import ChargeDistribution

_header = """
*********************************
|   UU  UU   EEEEEE    PPPP     |
|   UU  UU   EE        PP  PP   |
|   UU  UU   EEEEEE    PPPP     |
|   UU  UU   EE        PP       |
|    UUUU    EEEEEE    PP       |
*********************************

Unperturbed Electrostatic Potential
optimiser for mu+ stopping site finding

by Simone Sturniolo (2018)

Calculations started on {0}

""".format(datetime.datetime.now())


def geomopt(params, outf=None):

    t0 = datetime.datetime.now()

    chdistr = ChargeDistribution(seedname=params['chden_seed'],
                                 path=params['chden_path'],
                                 gw_fac=params['gw_factor'])

    a = chdistr.atoms
    m = params['particle_mass']

    if outf is not None:
        outf.write(_header)
        outf.write('Charge distribution loaded from {0}\n'.format(
            os.path.join(params['chden_path'], params['chden_seed'])))
        outf.write('Gaussian width factor used: {0}\n'.format(
            params['gw_factor']))
        outf.write('Particle mass: {0} kg\n'.format(m))

    # Objective functions
    def f(x):
        return chdistr.V([x])[0][0]

    def fprime(x):
        return chdistr.dV([x])[0][0]*1e-10

    def fhess(x):
        return chdistr.d2V([x])[0][0]*1e-20

    if outf is not None:
        outf.write('\n---------\n\n')
        outf.write('Performing optimisation with method {0}\n'.format(
            params['opt_method']))
        outf.write('Tolerance required for convergence: {0} eV\n'.format(
            params['opt_tol']))
        outf.write('Maximum number of steps: {0}\n'.format(
            params['geom_steps']))
        outf.write('Defect starting position: {0} {1} {2} Ang\n'.format(
            *params['mu_pos']))

    sol = minimize(f, params['mu_pos'], jac=fprime, hess=fhess,
                   tol=params['opt_tol'],
                   method=params['opt_method'],
                   options={'maxiter': params['geom_steps']})

    xsol = sol.x
    fxsol = np.linalg.solve(a.get_cell(complete=True),
                            xsol)
    Eclass = sol.fun
    # Zero point energy
    hess = fhess(sol.x)*1e20*cnst.e
    evals, _ = np.linalg.eigh(hess)
    if (evals < 0).any():
        # The minimum is not stable!
        Ezp = np.nan
    else:
        Ezp = np.sum(0.5*cnst.hbar*(evals/m)**0.5)/cnst.e
    Etot = Eclass + Ezp

    if outf is not None:
        outf.write('\n---------\n\n')
        outf.write('Optimisation stopped after {0} steps\n'.format(
            sol.nit))
        if not sol.success:
            outf.write('Optimisation failed:\n{0}\n\n'.format(sol.message))

        outf.write('\n\n')
        outf.write('Final coordinates: {0} {1} {2} Ang\n'.format(*xsol))
        outf.write('Final fractional coordinates: {0} {1} {2}\n'.format(
            *fxsol))
        outf.write('Classical energy: {0} eV\n'.format(Eclass))

        if np.isnan(Ezp):
            outf.write('Extremum is saddle point, impossible to compute '
                       'zero-point energy')
        else:
            outf.write('Zero-point energy: {0} eV\n'.format(Ezp))
            outf.write('Quantum total energy: {0} eV\n'.format(Etot))

        dt = datetime.datetime.now() - t0

        outf.write('\n\n')
        outf.write('Calculation time: {0} s'.format(dt.seconds +
                                                    dt.microseconds*1e-6))

    # Finally, return the solution's values
    results = {
        'x': xsol,
        'fx': fxsol,
        'Eclass': Eclass,
        'Ezp': Ezp,
        'Etot': Etot,
        'sol': sol,     # For any additional details about the calculation
        'struct': a     # For reference, the structure
    }

    return results


def geomopt_entry():
    parser = ap.ArgumentParser()
    parser.add_argument('input', type=str,
                        help="Input YAML file for the calculation")
    args = parser.parse_args()

    params = load_input_file(args.input, UEPOptSchema)

    seedpath = os.path.splitext(args.input)[0]
    seedname = os.path.split(seedpath)[1]

    if params['chden_seed'] is None:
        params['chden_seed'] = seedname  # Default is the same

    with open(seedpath + '.uep', 'w') as outf:
        results = geomopt(params, outf)

    # Now dump results
    if params['save_pickle']:
        pickle.dump(results, open(seedpath + '.uep.pkl', 'wb'))


if __name__ == "__main__":
    geomopt_entry()
