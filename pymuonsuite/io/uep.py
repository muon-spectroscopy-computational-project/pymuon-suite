# Python 2-to-3 compatibility code
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

# Write and read files for the UEP calculations

import os
import yaml
import pickle
import subprocess as sp
from ase import Atoms
from scipy.constants import physical_constants as pcnst


class ReadWriteUEP(object):

    def read(self, folder, sname=None, atoms=None, calc_type="muairss"):
        if sname is None:
            sname = os.path.split(folder)[-1]

        calc = UEPCalculator(label=sname, path=folder)

        try:
            calc.read()
        except Exception as e:
            print("Error: could not find UEP file in {0}".format(folder))
            return

        a = Atoms('H', positions=[calc._x_opt])

        if atoms is not None:
            a = atoms + a

        a.info['name'] = sname
        a.set_calculator(calc)
        calc.atoms = a

        return a

    def write(self, a, folder, params={}, script=None, calc=None, calc_type="muairss"):

        calc = self.create_calculator(a, folder, params, calc)

        calc.write_input()

        if script is not None:
            stxt = open(script).read()
            stxt = stxt.format(seedname=name)
            with open(os.path.join(folder, 'script.sh'), 'w') as sf:
                sf.write(stxt)

    def create_calculator(self, a, folder, params={}, calc=None):

        if not isinstance(calc, UEPCalculator):
            calc = UEPCalculator(atoms=a, chden=params['uep_chden'])
        else:
            dummy = UEPCalculator(chden=params['uep_chden'])
            calc.chden_path = dummy.chden_path
            calc.chden_seed = dummy.chden_seed

        if not params['charged']:
            raise RuntimeError("Can't use UEP method for neutral system")

        calc.label = params['name']
        calc.path = folder
        calc.gw_factor = params['uep_gw_factor']
        calc.geom_steps = params['geom_steps']
        calc.opt_tol = params['geom_force_tol']

        return calc


class UEPCalculator(object):
    """Mock 'calculator' used to store info to set up a UEP calculation"""

    def __init__(self, label='struct', atoms=None, index=-1, path='',
                 chden=''):

        self.label = label
        self.atoms = atoms
        self.index = index
        self.path = path

        chden = os.path.abspath(chden)
        chpath, chname = os.path.split(chden)
        chseed = os.path.splitext(chname)[0]

        self.chden_path = chpath
        self.chden_seed = chseed

        # Fixed parameters that can be changed later
        self.geom_steps = 30
        self.opt_tol = 1e-5
        self.gw_factor = 5.0
        self.opt_method = 'trust-exact'

        # Results
        self._Eclass = None
        self._Ezp = None
        self._Etot = None
        self._x_opt = None
        self._fx_opt = None

    @property
    def Eclass(self):
        self.read()
        return self._Eclass

    @property
    def Ezp(self):
        self.read()
        return self._Ezp

    @property
    def Etot(self):
        self.read()
        return self._Etot

    @property
    def x_opt(self):
        self.read()
        return self._x_opt

    @property
    def fx_opt(self):
        self.read()
        return self._fx_opt

    def get_potential_energy(self, a):
        return self._Eclass

    def write_input(self, a=None):

        if a is None:
            a = self.atoms

        if a is None:
            raise ValueError('Must pass one structure to write input')
        try:
            pos = a.get_positions()[self.index]
            mass = a.get_masses()[self.index]
        except IndexError as err:
            raise ValueError('Structure does not contain index of '
                             'UEPCalculator')

        outdata = {
            'mu_pos': list(map(float, pos)),
            'particle_mass': float(mass *
                                   pcnst['atomic mass constant'][0]),
            'chden_path': self.chden_path,
            'chden_seed': self.chden_seed,
            'geom_steps': self.geom_steps,
            'opt_tol': self.opt_tol,
            'opt_method': self.opt_method,
            'gw_factor': self.gw_factor,
            'save_pickle': True,                # Always save it with a "calculator"
        }

        yaml.dump(outdata, open(os.path.join(self.path, self.label + '.yaml'),
                                'w'))

    def run(self):

        self.write_input()

        proc = sp.Popen(['pm-uep-opt', '{0}.yaml'.format(self.label)],
                        cwd=os.path.abspath(self.path),
                        stdout=sp.PIPE, stderr=sp.PIPE)
        stdout, stderr = proc.communicate()

    def read(self):

        try:
            results = pickle.load(open(os.path.join(self.path,
                                                    self.label + '.uep.pkl'),
                                       'rb'))

            self._Eclass = results['Eclass']
            self._Ezp = results['Ezp']
            self._Etot = results['Etot']
            self._x_opt = results['x']
            self._fx_opt = results['fx']
        except FileNotFoundError:
            self.run()
            self.read()


def uep_write_input(a, folder, calc=None, name=None, script=None):

    if name is None:
        name = os.path.split(folder)[-1]

    if calc is None:
        calc = UEPCalculator()

    calc.path = folder
    calc.label = name
    calc.atoms = a

    calc.write_input()

    if script is not None:
        stxt = open(script).read()
        stxt = stxt.format(seedname=name)
        with open(os.path.join(folder, 'script.sh'), 'w') as sf:
            sf.write(stxt)


def uep_read_input(folder, name=None, atoms=None):

    if name is None:
        name = os.path.split(folder)[-1]

    calc = UEPCalculator(label=name, path=folder)
    calc.read()

    a = Atoms('H', positions=[calc._x_opt])

    if atoms is not None:
        a = atoms + a

    a.info['name'] = name
    a.set_calculator(calc)
    calc.atoms = a

    return a
