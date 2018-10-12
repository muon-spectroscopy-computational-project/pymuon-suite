# pymuon-suite
Collection of scripts and utilities for muon spectroscopy

## Command line scripts

The following is a list of currently supported command line scripts. For any
of them, use `<script> --help` to see usage information.

* `pm-muairss-gen`: generates a number of random muon structures for AIRSS
optimisation using Poisson spheres distribution and different calculators
* `pm-uep-opt`: Unperturbed Electrostatic Potential optimisation; may be 
unified with `pm-muairss-gen` in the future since it basically does the same
thing (but also carries out the calculation and extracts the results)
* `pm-nq`: generates input files for quantum effects using a phonon
approximation (work in progress)
