# Symmetry analysis: pm-symmetry

`pm-symmetry` analyses the symmetry of a structure with
[spglib](https://spglib.readthedocs.io/) and identifies the Wyckoff points,
which ones are occupied, and which ones can be uniquely identified as being
extrema rather than saddle points, thus providing some candidates for stopping
sites in crystals. Since the electrostatic potential is a smooth periodic
function of the crystal, its extrema tend to fall on high-symmetry points.

It is used as:

```
pm-symmetry <structure file> [-sp SYMPREC]
```

where `<structure file>` is any ASE-readable structure file.

Options:

* **-sp, --symprec** _(float)_: symmetry precision to use in spglib. Default
  is 1e-3.

The output is a report listing the space group of the structure and its
Wyckoff points, marking for each whether it is occupied by an atom and
whether the potential there is constrained by symmetry to be an extremum.
