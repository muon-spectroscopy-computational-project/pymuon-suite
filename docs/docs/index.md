# pymuon-suite

`pymuon-suite` is a collection of scripts and utilities for muon spectroscopy,
developed by the [Muon Spectroscopy Computational Project](https://muon-spectroscopy-computational-project.github.io/index.html).
It is designed to help answer questions such as: what are the optimal stopping
sites for a muon in a certain crystalline solid? What are the effects of
quantum delocalisation on the hyperfine coupling of the muon?

All tools are command-line scripts driven by YAML parameter files, which are
validated at startup. Structure files are read with the
[Atomic Simulation Environment (ASE)](https://docs.ase-lib.org/), so any
ASE-readable format (CIF, CASTEP `.cell`, XYZ, ...) is accepted.

## Command line tools

| Command | Purpose |
|---|---|
| `pm-muairss` | Generate random muon insertion sites for AIRSS-style optimisation with CASTEP, DFTB+ or UEP, and cluster the results afterwards |
| `pm-muairss-gen` | Alias for `pm-muairss -t w` (generation only) |
| `pm-uep-opt` | Unperturbed Electrostatic Potential (UEP) optimisation of a single muon in a unit cell |
| `pm-uep-plot` | Plot the UEP along lines or planes of a unit cell |
| `pm-nq` | Generate input files for, or analyse the results of, nuclear quantum effects calculations in the phonon approximation |
| `pm-asephonons` | Compute phonons with ASE and DFTB+ for use with `pm-nq` |
| `pm-symmetry` | Symmetry analysis of a structure: Wyckoff points and candidate stopping sites |

For any of them, use `<script> --help` to see usage information.

## Typical workflows

**Classical muon site search (AIRSS):** generate many random muon starting
positions with [`pm-muairss`](./muairss.md), relax each one with CASTEP, DFTB+
or [UEP](./uep.md), then run `pm-muairss` again to cluster the relaxed
structures into candidate stopping sites.

**UEP-only site search:** if a CASTEP charge density of the host material is
available, [`pm-uep-opt`](./uep.md) finds candidate sites by minimising the
unperturbed electrostatic potential — no further DFT calculations needed.

**Quantum corrections:** starting from a relaxed muon site, use
[`pm-nq`](./quantum.md) to displace the muon along its phonon modes, run DFT
on the displaced structures, and average properties such as hyperfine
couplings over the quantum distribution.

## Citing pymuon-suite

If you use `pymuon-suite` in your research, please cite both the software
itself and the relevant papers depending on which tools you used (see also
[`CITATION.cff`](https://github.com/muon-spectroscopy-computational-project/pymuon-suite/blob/main/CITATION.cff)
in the repository):

**The software** (concept DOI covering all versions):

> Sturniolo, S., Liborio, L., Chadwick, E., Murgatroyd, L., Laverack, A.,
> Mudaraddi, A., Austin, P., Davies, J., Muon Spectroscopy Computational
> Project, *pymuon-suite*,
> [doi:10.5281/zenodo.7025643](https://doi.org/10.5281/zenodo.7025643)

**For `pm-muairss` and `pm-muairss-gen`:**

> S. Sturniolo, L. Liborio, S. Jackson,
> "Comparison between density functional theory and density functional tight
> binding approaches for finding the muon stopping site in organic molecular
> crystals", J. Chem. Phys. 150, 154301 (2019),
> [doi:10.1063/1.5085197](https://doi.org/10.1063/1.5085197)

**For `pm-uep-opt`, `pm-uep-plot` and `pm-symmetry`:**

> S. Sturniolo, L. Liborio,
> "Computational prediction of muon stopping sites: A novel take on the
> unperturbed electrostatic potential method",
> J. Chem. Phys. 153, 044111 (2020),
> [doi:10.1063/5.0012381](https://doi.org/10.1063/5.0012381)
