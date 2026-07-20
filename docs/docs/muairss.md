# Muon site search: pm-muairss

`pm-muairss` carries out the two halves of an AIRSS-style muon site search:

1. **Write** (`-t w`): generate a number of random starting structures with a
   muon inserted, to then converge with an outside calculator (CASTEP, DFTB+
   or [UEP](./uep.md)) to potential stopping sites;
2. **Read** (`-t r`, the default): once those calculations are done, read the
   relaxed structures back and cluster them into candidate stopping sites.

The script is used as follows:

```
pm-muairss <structure> <parameter file> [-t r|w]
```

Here `<structure>` may be a single structure file in a format readable by the
Atomic Simulation Environment (ASE), or a folder containing multiple ones, in
which case separate folders will be generated for the output structures.
`<parameter file>` is a file in YAML format containing the parameters defining
the way the structures are generated. If a folder is passed as the first
argument, the parameters can be overridden by structure-specific YAML files.

The default task is READ; this split is meant to help avoid overwriting one's
results by mistake. `pm-muairss-gen` is an alias for `pm-muairss` with the
`-t w` option on.

A typical YAML parameter file may look like:

```yaml
name: structure
poisson_r: 0.8
supercell: [2, 2, 2]
```

with variables written like Python values (strings, floats, or lists). The
important thing is to make sure that the separators after the colon are spaces
and not tabs, since the latter are not compatible with the format. Here we
present a list of acceptable keywords to use in this file.

## General keywords

* **name** _(string)_: name to call the folder for containing each structure.
  This name will be postfixed with a unique number, e.g. `struct_001`. Default
  is `struct`.
* **calculator** _(string or list of strings)_: calculator to generate
  structure files for. Must be a single word or a comma separated list of
  values. Currently supported calculators are CASTEP, DFTB+ and UEP. Can also
  pass `all` as an option to generate files for all calculators. Default is
  `dftb+`.
* **script\_file** _(string)_: path to script file to copy in all folders.
* **particle\_mass\_amu** _(float)_: mass of the added particle in amu.
  Defaults to the muon mass.
  _Warning:_ this parameter _only_ modifies the mass. If you wish to model a
  non-muon particle, other physical parameters may also be different (e.g.
  spin), and not all of these are customisable at present. This may generate
  inaccurate results, so use this parameter at your own risk. Please
  [raise an issue](https://github.com/muon-spectroscopy-computational-project/pymuon-suite/issues)
  on the pymuon-suite repository to request support for additional parameters.
* **poisson\_r** _(float)_: Poisson sphere radius to use for random
  generation. No two starting muon positions will be closer than this
  distance. Smaller values make for bigger structure sets. Default is 0.8 Å.
* **vdw\_scale** _(float)_: Van der Waals scaling factor to use when
  generating muon sites to avoid existing atoms. Smaller values will allow
  muons to get closer to the other ions. Default is 0.5.
* **charged** _(bool)_: if True, the muon will be considered charged instead
  of a muonium with an accompanying electron. Must be True for UEP
  calculations. Default is False.
* **supercell** _(int or list of ints)_: supercell size and shape to use. This
  can either be a single int, a list of three integers or a 3x3 matrix of
  integers. For a single number a diagonal matrix will be generated with the
  integer repeated on the diagonals. For a list of three numbers a diagonal
  matrix will be generated where the diagonal elements are set to the list. A
  matrix will be used directly as is. Default is a 3x3 identity matrix.
* **k\_points\_grid** _(list of ints)_: list of three integer k-points.
  Default is `[1,1,1]`.
* **out\_folder** _(string)_: name to call the output folder used to store the
  input files that the script generates. Default is `./muon-airss-out`.
* **mu\_symbol** _(string)_: the symbol to use for the muon when writing out
  the CASTEP custom species. Must use the format `"X:custom"` where `X` is an
  element symbol and `custom` can be any string. Default is `H:mu`.
* **geom\_steps** _(int)_: maximum number of geometry optimisation steps.
* **geom\_force\_tol** _(float)_: tolerance on geometry optimisation in units
  of eV/Å.
* **random\_seed** _(int)_: random seed for the structure generation, for
  reproducibility.

## CASTEP keywords

* **castep\_command** _(string)_: command to use to run CASTEP. Default is
  `castep.serial`.
* **castep\_param** _(string)_: file path to the CASTEP parameter file.

## DFTB+ keywords

* **dftb\_command** _(string)_: command to use to run DFTB+. Default is
  `dftb+`.
* **dftb\_set** _(string)_: the parameter set to use for DFTB+. Currently
  supported are `3ob-3-1` and `pbc-0-3`. For more information see
  [the DFTB site](https://dftb.org/parameters/).
* **dftb\_optionals** _(list of strings)_: additional optional json files to
  activate for DFTBArgs (for example, `dftd3.json` will use DFTD3 dispersion
  forces for `3ob-3-1` if DFTB+ has been compiled to support them).
* **dftb\_pbc** _(bool)_: whether to turn on periodic boundary conditions in
  DFTB+. Default is True.
* **max\_scc\_steps** _(int)_: if applicable, max number of SCC steps to
  perform before giving up. Default is 200 which is also the default for
  DFTB+.

## UEP keywords

See the [UEP page](./uep.md) for details on the method.

* **uep\_chden** _(string)_: path to the `.den_fmt` file containing the
  electronic density for an Unperturbed Electrostatic Potential optimisation.
  The corresponding `.castep` file must be in the same folder and with the
  same seedname.
* **uep\_gw\_factor** _(float)_: Gaussian width factor for UEP calculation.
  Higher values will make the potential of atomic nuclei closer to the
  point-like approximation but may introduce artifacts. Default is 5.0.
* **uep\_save\_structs** _(bool)_: save a structure file for each optimised
  structure + muon in `.xyz` format when running UEP. Default is True.

## Clustering keywords

These control the analysis performed in the READ (`-t r`) step:

* **clustering\_method** _(string)_: clustering method to use, either `hier`
  (hierarchical) or `kmeans` (k-means). Default is `hier`.
* **clustering\_hier\_t** _(float)_: `t` parameter for hierarchical
  clustering. Default is 0.3.
* **clustering\_kmeans\_k** _(int)_: number of clusters for k-means
  clustering. Default is 4.
* **clustering\_save\_type** _(string)_: whether to save, for each cluster,
  the minimum energy `structures`, or `input` files for another calculator.
* **clustering\_save\_format** _(string)_: if `clustering_save_type` is
  `structures`, this takes a file extension (`cif`, `xyz`, etc.); if `input`,
  it takes the name of a program (`castep`, `dftb+`, etc.).
* **clustering\_save\_folder** _(string)_: name of the folder used to store
  the files selected with the two options above.
* **allpos\_filename** _(string)_: name of a file to save with all muon
  positions in one.
* **clustering\_save\_min** _(bool)_: _deprecated_ — replaced by
  `clustering_save_type` and `clustering_save_format`.
