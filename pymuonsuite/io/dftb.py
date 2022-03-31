import os
import pickle
import glob
import warnings
import numpy as np

from copy import deepcopy

from soprano.utils import customize_warnings, silence_stdio

from ase import io
from ase.calculators.dftb import Dftb
from ase.calculators.singlepoint import SinglePointCalculator

from pymuonsuite.utils import BackupFile
from pymuonsuite.calculate.hfine import compute_hfine_mullpop
from pymuonsuite import constants

from pymuonsuite.io.readwrite import ReadWrite

_geom_opt_args = {
    "Driver_": "ConjugateGradient",
    "Driver_Masses_": "",
    "Driver_Masses_Mass_": "",
    "Driver_Masses_Mass_Atoms": "-1",
    "Driver_Masses_Mass_MassPerAtom [amu]": str(constants.m_mu_amu),
}

_spinpol_args = {
    "Hamiltonian_SpinPolarisation_": "Colinear",
    "Hamiltonian_SpinPolarisation_UnpairedElectrons": 1,
    "Hamiltonian_SpinPolarisation_InitialSpins_": "",
    "Hamiltonian_SpinPolarisation_InitialSpins_Atoms": "-1",
    "Hamiltonian_SpinPolarisation_InitialSpins_SpinPerAtom": 1,
}

customize_warnings()


class ReadWriteDFTB(ReadWrite):
    def __init__(self, params={}, script=None, calc=None):
        """
        |   Args:
        |   params (dict):          Contains dftb_set, k_points_grid,
        |                           geom_force_tol and dftb_optionals.
        |                           geom_steps, max_scc_steps, and
        |                           charged are also required in the case
        |                           of writing geom_opt input files
        |   script (str):           Path to a file containing a submission
        |                           script to copy to the input folder. The
        |                           script can contain the argument
        |                           {seedname} in curly braces, and it will
        |                           be appropriately replaced.
        |   calc (ase.Calculator):  Calculator to attach to Atoms. If
        |                           present, the pre-existent one will
        |                           be ignored.
        """

        self.set_params(params)
        self.set_script(script)
        self._calc = calc
        self._calc_type = None

    def set_params(self, params):
        """
        |   Args:
        |   params (dict):          Contains dftb_set, k_points_grid,
        |                           geom_force_tol and dftb_optionals.
        |                           geom_steps, max_scc_steps, and
        |                           charged are also required in the case
        |                           of writing geom_opt input files
        """
        if not (isinstance(params, dict)):
            raise ValueError("params should be a dict, not ", type(params))
            return

        self.params = deepcopy(params)
        # resetting this to None makes sure that the calc is recreated after
        # the params are updated:
        self._calc_type = None

    def read(
        self, folder, sname=None, read_spinpol=False, read_phonons=False, **kwargs
    ):
        """Read a DFTB+ output non-destructively.
        |
        |   Args:
        |   folder (str) :          path to a directory to load DFTB+ results
        |   sname (str):            name to label the atoms with and/or of the
        |                           .phonons.pkl file to be read
        |   Returns:
        |   atoms (ase.Atoms):      an atomic structure with the results
        |                           attached in a SinglePointCalculator
        """

        try:
            with silence_stdio():
                atoms = io.read(os.path.join(folder, "geo_end.gen"))

        except IOError:
            raise IOError(
                "ERROR: No geo_end.gen file found in {}.".format(
                    os.path.abspath(folder)
                )
            )
        except Exception as e:
            raise IOError(
                "ERROR: Could not read {file}, due to error: {error}".format(
                    file="geo_end.gen", error=e
                )
            )
        if sname is None:
            atoms.info["name"] = os.path.split(folder)[-1]
        else:
            atoms.info["name"] = sname
        results_file = os.path.join(folder, "results.tag")
        if os.path.isfile(results_file):
            # DFTB+ was used to perform the optimisation
            temp_file = os.path.join(folder, "results.tag.bak")

            # We need to backup the results file here because
            # .read_results() will remove the results file
            with BackupFile(results_file, temp_file):
                calc = Dftb(atoms=atoms)
                calc.atoms_input = atoms
                calc.directory = folder
                calc.do_forces = True
                calc.read_results()

            energy = calc.get_potential_energy()
            forces = calc.get_forces()
            charges = calc.get_charges(atoms)

            calc = SinglePointCalculator(
                atoms, energy=energy, forces=forces, charges=charges
            )

            atoms.calc = calc

        if read_spinpol:
            try:
                pops = parse_spinpol_dftb(folder)
                hfine = []
                for i in range(len(atoms)):
                    hf = compute_hfine_mullpop(
                        atoms, pops, self_i=i, fermi=True, fermi_neigh=True
                    )
                    hfine.append(hf)
                atoms.set_array("hyperfine", np.array(hfine))
            except (IndexError, IOError) as e:
                raise IOError(
                    "Could not read hyperfine details due to error: " "{0}".format(e)
                )

        if read_phonons:
            try:
                if sname is not None:
                    phonon_source_file = os.path.join(folder, sname + ".phonons.pkl")
                else:
                    print(
                        "Phonons filename was not given, searching for any"
                        " .phonons.pkl file."
                    )
                    phonon_source_file = glob.glob(
                        os.path.join(folder, "*.phonons.pkl")
                    )[0]
                self._read_dftb_phonons(atoms, phonon_source_file)
            except IndexError:
                raise IOError(
                    "No .phonons.pkl files found in {}.".format(os.path.abspath(folder))
                )
            except IOError:
                raise IOError("{} could not be found.".format(phonon_source_file))
            except Exception as e:
                raise IOError(
                    "Could not read {file} due to error: {error}".format(
                        file=phonon_source_file, error=e
                    )
                )

        return atoms

    def _read_dftb_phonons(self, atoms, phonon_source_file):
        with open(phonon_source_file, "rb") as f:
            phdata = pickle.load(f)
            # Find the gamma point
            gamma_i = None
            for i, p in enumerate(phdata.path):
                if (p == 0).all():
                    gamma_i = i
                    break
            try:
                ph_evals = phdata.frequencies[gamma_i]
                ph_evecs = phdata.modes[gamma_i]
                atoms.info["ph_evals"] = ph_evals
                atoms.info["ph_evecs"] = ph_evecs
            except TypeError:
                raise RuntimeError(
                    ("Phonon file {0} does not contain gamma " "point data").format(
                        phonon_source_file
                    )
                )

    def write(self, a, folder, sname=None, calc_type="GEOM_OPT"):
        """Writes input files for an Atoms object with a Dftb+
        calculator.

        | Args:
        |   a (ase.Atoms):          Atoms object to write. Can have a Dftb
        |                           calculator attached to carry
        |                           arguments.
        |   folder (str):           Path to save the input files to.
        |   sname (str):            Seedname to save the files with. If not
        |                           given, use the name of the folder.
        |   calc_type (str):        Calculation which will be performed:
        |                           "GEOM_OPT" or "SPINPOL"
        """

        if calc_type == "GEOM_OPT" or calc_type == "SPINPOL":
            if sname is None:
                sname = os.path.split(folder)[-1]  # Same as folder name

            if self._calc is None and isinstance(a.calc, Dftb):
                self._calc = a.calc

            self._calc = deepcopy(self._calc)

            # only create a new calc if the calc type requested is different
            # to that already saved.
            if calc_type != self._calc_type:
                self._create_calculator(calc_type=calc_type)

            a.calc = self._calc
            a.calc.label = sname
            a.calc.directory = folder
            a.calc.write_input(a)

            if self.script is not None:
                stxt = open(self.script).read()
                stxt = stxt.format(seedname=sname)
                with open(os.path.join(folder, "script.sh"), "w") as sf:
                    sf.write(stxt)
        else:
            raise (
                NotImplementedError(
                    "Calculation type {} is not implemented."
                    " Please choose 'GEOM_OPT' or 'SPINPOL'".format(calc_type)
                )
            )

    def _create_calculator(self, calc_type="GEOM_OPT"):
        from pymuonsuite.data.dftb_pars.dftb_pars import DFTBArgs

        if isinstance(self._calc, Dftb):
            calc_kpts = deepcopy(self._calc.kpts)
            args = self._calc.todict()
        else:
            calc_kpts = None
            args = {}

        # We set kpoints if dftb_pbc has been set to true,
        # or there are kpoints set in the provided calc:
        if self.params.get("dftb_pbc"):
            if self.params.get("k_points_grid") is None:
                self.params["k_points_grid"] = calc_kpts
                if self.params.get("k_points_grid") is None:
                    self.params["k_points_grid"] = np.ones(3).astype(int)
            self._calc = Dftb(kpts=self.params["k_points_grid"])
        else:
            self.params["k_points_grid"] = calc_kpts
            if self.params.get("k_points_grid") is None:
                self._calc = Dftb()
            else:
                self._calc = Dftb(kpts=self.params["k_points_grid"])

        dftb_set_param = self.params.get("dftb_set", "3ob-3-1")

        dargs = DFTBArgs(dftb_set_param)

        if "dftb_optionals" not in self.params:
            self.params["dftb_optionals"] = []

        if calc_type == "SPINPOL":
            self.params["dftb_optionals"].append("spinpol.json")

        for opt in self.params["dftb_optionals"]:
            try:
                dargs.set_optional(opt, True)
            except KeyError:
                if opt == "spinpol.json":
                    raise ValueError(
                        "DFTB+ parameter set does not allow spin"
                        "polarised calculations"
                    )
                else:
                    warnings.warn(
                        "Warning: optional DFTB+ file {0} not"
                        "available for {1}"
                        " parameter set, skipping"
                    ).format(opt, self.params["dftb_set"])
        args.update(dargs.args)

        if calc_type == "GEOM_OPT":
            args.update(_geom_opt_args)

            # If the following parameters are set in the params dict we take
            # their values from there.
            # Otherwise, we take their values from the calculator that has
            # been provided.
            # If neither of these have been set, we use the default values.

            charge_param = self.params.get("charged")
            if charge_param is not None:
                args["Hamiltonian_Charge"] = 1.0 * charge_param
            else:
                if args.get("Hamiltonian_Charge") is None:
                    args["Hamiltonian_Charge"] = 0.0

            particle_mass_param = self.params.get("particle_mass_amu")

            if particle_mass_param is not None:
                args["Driver_Masses_Mass_MassPerAtom [amu]"] = particle_mass_param
            elif args.get("Driver_Masses_Mass_MassPerAtom [amu]") is None:
                args["Driver_Masses_Mass_MassPerAtom [amu]"] = constants.m_mu_amu

            geom_steps_param = self.params.get("geom_steps")

            if geom_steps_param is not None:
                args["Driver_MaxSteps"] = geom_steps_param
            else:
                if args.get("Driver_MaxSteps") is None:
                    args["Driver_MaxSteps"] = 30

            geom_force_tol_param = self.params.get("geom_force_tol")

            if geom_force_tol_param is not None:
                args["Driver_MaxForceComponent [eV/AA]"] = geom_force_tol_param
            else:
                if args.get("Driver_MaxForceComponent [eV/AA]") is None:
                    args["Driver_MaxForceComponent [eV/AA]"] = 0.05

            max_scf_cycles_param = self.params.get("max_scc_steps")
            if max_scf_cycles_param is not None:
                args["Hamiltonian_MaxSccIterations"] = max_scf_cycles_param
            else:
                if args.get("Hamiltonian_MaxSccIterations") is None:
                    args["Hamiltonian_MaxSccIterations"] = 200

        elif calc_type == "SPINPOL":
            del args["Hamiltonian_SpinPolarisation"]
            args.update(_spinpol_args)
            self._calc.do_forces = True
        else:
            raise (
                NotImplementedError(
                    "Calculation type {} is not implemented."
                    " Please choose 'GEOM_OPT' or 'SPINPOL'".format(calc_type)
                )
            )

        self._calc.parameters.update(args)

        self._calc_type = calc_type

        return self._calc


def parse_spinpol_dftb(folder):
    """Parse atomic spin populations from a detailed.out DFTB+ file."""

    with open(os.path.join(folder, "detailed.out")) as f:
        lines = f.readlines()

    # Find the atomic populations blocks
    spinpol = {
        "up": [],
        "down": [],
    }

    charges = {}

    for i, l in enumerate(lines):
        if "Atomic gross charges (e)" in l:
            for ll in lines[i + 2 :]:
                lspl = ll.split()[:2]
                try:
                    a_i, q = int(lspl[0]), float(lspl[1])
                except (IndexError, ValueError):
                    break
                charges[a_i - 1] = q

        if "Orbital populations" in l:
            s = l.split()[2][1:-1]
            if s not in spinpol:
                raise RuntimeError("Invalid detailed.out file")
            for ll in lines[i + 2 :]:
                lspl = ll.split()[:5]
                try:
                    a_i, n, l, m, pop = map(float, lspl)
                except ValueError:
                    break
                a_i, n, l, m = map(int, [a_i, n, l, m])
                if len(spinpol[s]) < a_i:
                    spinpol[s].append({})
                spinpol[s][a_i - 1][(n, l, m)] = pop

    # Build population and net spin
    N = len(spinpol["up"])
    if N == 0:
        raise RuntimeError("No atomic populations found in detailed.out")

    pops = [{} for i in range(N)]

    # Start with total populations and total spin
    for i in range(N):
        pops[i] = {
            "q": charges[i],
            "pop": 0,
            "spin": 0,
            "pop_orbital": {},
            "spin_orbital": {},
        }
        for s, sign in {"up": 1, "down": -1}.items():
            for nlm, p in spinpol[s][i].items():
                pops[i]["pop"] += p
                pops[i]["spin"] += sign * p
                pops[i]["pop_orbital"][nlm] = pops[i]["pop_orbital"].get(nlm, 0.0) + p
                pops[i]["spin_orbital"][nlm] = (
                    pops[i]["spin_orbital"].get(nlm, 0.0) + p * sign
                )

    return pops
