[![Build Status](https://github.com/muon-spectroscopy-computational-project/pymuon-suite/actions/workflows/python-package-tests.yml/badge.svg?branch=main)](https://github.com/muon-spectroscopy-computational-project/pymuon-suite/actions/workflows/python-package-tests.yml)
[![Codecov](https://codecov.io/gh/muon-spectroscopy-computational-project/pymuon-suite/branch/main/graph/badge.svg)](https://codecov.io/gh/muon-spectroscopy-computational-project/pymuon-suite)

# pymuon-suite
Collection of scripts and utilities for muon spectroscopy.

## Installation

Requires Python 3.10+. Install with pip or conda:

`pip install pymuonsuite`

`conda install pymuonsuite`

If you plan to read CASTEP .phonon files, you will also need to install
[Euphonic](https://github.com/pace-neutrons/Euphonic):

`pip install euphonic`

`conda install euphonic`

Help with Euphonic installation can be found in the
[Euphonic documentation](https://euphonic.readthedocs.io/en/latest/installation.html).

### `spglib` build failed during pip install

On some platforms, additional tools are needed to build the `spglib` Python module when installing
via pip. On Windows, you may need to install
[Microsoft C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/); on Linux
you may need to `apt-get install python-dev` or `yum install python-devel` according to your
distribution. This should not be necessary if installing via conda, and so we recommend using 
conda if you want to avoid installing these tools.

Further help with Spglib installation can be found in the
[Spglib documentation](https://spglib.github.io/spglib/python-spglib.html#installation).


## Command line scripts

The following is a list of currently supported command line scripts. For any
of them, use `<script> --help` to see usage information.

* `pm-muairss`: generates a number of random muon structures for AIRSS
optimisation using Poisson spheres distribution and different calculators, as well as carries out their clustering analysis after the calculations have been done. Usage is `pm-muairss <structure file> <parameter file>`, with the additional option `-t w` when one desires to generate the structures instead of analysing the results. This is done to help avoid overwriting one's results by mistake;
* `pm-muairss-gen`: alias for `pm-muairss` with the `-t w` option on;
* `pm-uep-opt`: Unperturbed Electrostatic Potential optimisation for a single muon in a unit cell; it's used as `pm-uep-opt <parameter file>`;
* `pm-uep-plot`: Unperturbed Electrostatic Potential plotting for a given unit cell and specific lines or planes along it; it's used as `pm-uep-plot <parameter file>`;
* `pm-symmetry`: analyses the symmetry of a structure with `spglib` and identifies the Wyckoff points, which ones are occupied, and which ones can be uniquely identified as being extrema rather than saddle points, thus providing some candidates for stopping sites in crystals; it's used as `pm-symmetry <structure file>`;
* `pm-asephonons`: compute phonons for the given structure using ASE and DFTB+;
* `pm-nq`: generates input files for quantum effects using a phonon
approximation or analyses the results (work in progress)

For more in-depth information about each tool and their usage, [check the Wiki](https://github.com/muon-spectroscopy-computational-project/pymuon-suite/wiki).
