# Quantum effects: pm-nq and pm-asephonons

The muon is a very light particle, and its zero-point motion can be
significant. `pm-nq` estimates nuclear quantum effects on muon-related
properties in the phonon (harmonic) approximation: starting from a relaxed
muon site with known phonon modes, it generates structures with the muon
displaced along its three phonon modes, lets an external code (CASTEP or
DFTB+) compute a property (e.g. hyperfine coupling tensors) for each displaced
structure, then averages that property over the quantum thermal distribution.

`pm-asephonons` is a companion tool that computes the phonon modes with ASE
and DFTB+, for use when a CASTEP `.phonon` file is not available.

## pm-nq

Like `pm-muairss`, `pm-nq` works in two steps, selected with the `-t` option:

```
pm-nq <structure> <parameter file> [-t r|w]
```

with `-t w` generating and WRITING the displaced structures, and `-t r`
(the default) READING and analysing the results once the external calculations
are done. `<structure>` is a structure file in an ASE readable format
containing the relaxed structure with the muon.

Reading CASTEP `.phonon` files requires
[Euphonic](https://github.com/pace-neutrons/Euphonic) to be installed (see
[Installation](./installation.md)).

The YAML parameter file accepts the following keywords:

* **method** _(string)_: method used to calculate the thermal average. Either
  `independent` (displace the muon along each phonon mode independently) or
  `montecarlo` (sample random displacements from the thermal distribution).
  Default is `independent`.
* **mu\_index** _(int)_: index of the muon in the cell. Default is -1 (the
  last atom).
* **mu\_symbol** _(string)_: if using CASTEP custom species, custom species of
  the muon (supersedes `mu_index` if present in the cell). Default is `H:mu`.
* **grid\_n** _(int)_: number of grid points to use on each phonon mode or
  pairs of thermal lines. Default is 20.
* **sigma\_n** _(float)_: number of sigmas to sample in the harmonic
  approximation. Default is 3.
* **k\_points\_grid** _(list of ints)_: list of three integer k-points for
  both phonon and hyperfine calculations. Default is `[1,1,1]`.
* **avgprop** _(string)_: property to be averaged; currently `hyperfine`
  (hyperfine coupling tensors) or `charge`. Default is `hyperfine`.
* **calculator** _(string)_: calculator to use for the property, `castep` or
  `dftb+`. Default is `castep`.
* **phonon\_source\_file** _(string)_: source file for the phonon modes (e.g.
  a CASTEP `.phonon` file, or the output of `pm-asephonons`).
* **phonon\_source\_type** _(string)_: type of source file for the phonon
  modes, `castep` or `dftb+`. Default is `castep`.
* **displace\_T** _(float)_: temperature (K) for displacement generation.
  Default is 0.
* **average\_T** _(float)_: temperature (K) for averaging. Defaults to the
  value of `displace_T`.
* **write\_allconf** _(bool)_: write a 'collective' file with all displaced
  positions in one. Default is False.
* **script\_file** _(string)_: path to script file to copy in all folders.
* **castep\_param** _(string)_: path of a CASTEP parameter file which can be
  copied into the folders with displaced cell files for convenience.
* **dftb\_pbc** _(bool)_: whether to turn on periodic boundary conditions in
  DFTB+. Default is True.
* **dftb\_set** _(string)_: if using DFTB+, which parametrisation to use,
  `3ob-3-1` or `pbc-0-3`. Default is `3ob-3-1`.
* **average\_file** _(string)_: name of the output file for the averages.
  Default is `averages.dat`.
* **random\_seed** _(int)_: random seed for the displacement generation, for
  reproducibility.

## pm-asephonons

Computes phonon modes with ASE and DFTB+ for reuse in quantum effects
calculations. It is used as:

```
pm-asephonons <structure file> <parameter file>
```

The YAML parameter file accepts the following keywords:

* **name** _(string)_: name to use for the output files. Defaults to the
  structure file's name.
* **phonon\_kpoint\_grid** _(list of ints)_: list of three integer k-points at
  which to compute the phonons. Default is `[1,1,1]`.
* **kpoint\_grid** _(list of ints)_: list of three integer k-points used for
  the DFTB+ calculation. Default is `[1,1,1]`.
* **force\_tol** _(float)_: force tolerance for the geometry optimisation.
  Default is 0.01.
* **dftb\_set** _(string)_: which DFTB+ parametrisation to use, `3ob-3-1` or
  `pbc-0-3`. Default is `3ob-3-1`.
* **pbc** _(bool)_: whether to turn on periodic boundary conditions. Default
  is True.
* **force\_clean** _(bool)_: force clean existing phonon files of the same
  name. Default is False.
