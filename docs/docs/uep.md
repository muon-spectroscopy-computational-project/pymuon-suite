# UEP: pm-uep-opt and pm-uep-plot

The Unperturbed Electrostatic Potential (UEP) method finds candidate muon
stopping sites by minimising the electrostatic potential computed from the
host crystal's charge density, as obtained from a previous CASTEP calculation
— no DFT geometry relaxation of the muonated structure is needed. Since the
muon is treated as a positive point charge and the host's electronic structure
is not allowed to relax around it, the method is only appropriate for the
diamagnetic muon state (a "naked", charged muon rather than muonium).

Both tools need the charge density from CASTEP in formatted form: a
`<seedname>.den_fmt` file, with the corresponding `<seedname>.castep` file in
the same folder.

For the theory behind the method, see the paper referenced on the
[home page](./index.md#citing-pymuon-suite):
S. Sturniolo, L. Liborio, J. Chem. Phys. 153, 044111 (2020),
[doi:10.1063/5.0012381](https://doi.org/10.1063/5.0012381).

## Shared keywords

Both `pm-uep-opt` and `pm-uep-plot` take a YAML parameter file accepting the
following keywords:

* **chden\_path** _(string)_: path from which to load the charge density
  files. Default is the current folder.
* **chden\_seed** _(string)_: seedname of the charge density calculation (the
  files `<chden_path>/<chden_seed>.den_fmt` and
  `<chden_path>/<chden_seed>.castep` must exist).
* **gw\_factor** _(float)_: Gaussian width factor for the ionic potential.
  Higher values will make the potential of atomic nuclei closer to the
  point-like approximation but may introduce artifacts. Default is 5.0.

## pm-uep-opt

Optimises the position of a single muon in the unit cell by following the
gradient of the UEP downhill to a minimum. It is used as:

```
pm-uep-opt <parameter file>
```

In addition to the shared keywords above, it accepts:

* **mu\_pos** _(list of three floats)_: starting position for the muon, in
  absolute (Cartesian) coordinates. Default is `[0, 0, 0]`.
* **geom\_steps** _(int)_: maximum number of geometry optimisation steps.
  Default is 30.
* **opt\_tol** _(float)_: tolerance on the optimisation. Default is 1e-5.
* **opt\_method** _(string)_: optimisation method, passed to
  `scipy.optimize.minimize`. Default is `trust-exact`.
* **particle\_mass** _(float)_: mass of the particle, in kg. Defaults to the
  muon mass.
* **save\_pickle** _(bool)_: whether to save the full result of the
  optimisation as a pickled `.uep` file. Default is True.
* **save\_structs** _(bool)_: whether to save a structure file for the
  optimised structure + muon in `.xyz` format. Default is True.

## pm-uep-plot

Plots the UEP for a given unit cell along specific lines or planes, writing
data files suitable for plotting (e.g. with gnuplot). It is used as:

```
pm-uep-plot <parameter file>
```

In addition to the shared keywords above, it accepts:

* **line\_plots** _(list)_: specifications for line plots. Each line can be
  given in one of the following formats:
    * `[[crystallographic direction], [starting point], length, number of points]`
    * `[[starting point], [end point], number of points]`
    * `[starting atom, end atom, number of points]`
* **plane\_plots** _(list)_: specifications for plane plots. Each plane can be
  given in one of the following formats:
    * `[[corner 1], [corner 2], [corner 3], points along width, points along height]`
    * `[corner atom 1, corner atom 2, corner atom 3, points along width, points along height]`

Atoms are referenced by their (0-based) index in the structure; points and
directions are vectors in fractional coordinates.
