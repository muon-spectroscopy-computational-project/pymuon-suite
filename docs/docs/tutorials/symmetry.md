## Symmetry analysis method

Symmetry considerations are important for determining the muon stopping sites. In crystallography, the sites with high symmetry are often the preferred atomic sites. This holds true for muons as well, and in practice, very often, the muon stopping site is an interstitial high-symmetry site of the crystal. For this reason, a crystallographic analysis of the pure system that will be studied with muons is a useful first step when looking for muon stopping sites, as such analysis is significantly faster than any computer simulation. 

A detailed description of the methodology is given in this [paper by S. Sturniolo and L. Liborio](https://aip.scitation.org/doi/10.1063/5.0012381). 

## Running the analysis

Once `pymuon-suite` is [installed](../installation.md), it provides the user both with a Python API to use for custom programs and with a tailored `pm-symmetry` script, which can be used to perform the symmetry analysis.

The symmetry script can be run simply by executing the command:

`pm-symmetry` `<structure file>`

where the structure file has to be any supported crystallographic file format (such as .cif or the CASTEP structure file .cell).  

### Example of application of the symmetry method

#### TiO<sub>2</sub> rutile 

The muon stopping sites in TiO<sub>2</sub> rutile were determined by transverse field μSR measurements performed in the MuSR instrument at ISIS (UK). In these stopping sites, the muon has a low temperature ground state and a high temperature excited state, both corresponding to a muon bound to one of the six oxygen atoms that form an octahedron around the Ti<sup>3+</sup> at the center of the TiO<sub>2</sub> rutile unit cell. Each one of these stopping sites has a different O–Ti<sup>3+</sup> bonding configuration, with the ground state formed by bonding the muon to the in-plane oxygen atoms that lie in the same plane as Ti<sup>3+</sup>. These two sites are related by symmetry and are only distinguished by the electronic structure of the TiO<sub>2</sub> rutile. 

Running `pm-symmetry` on a CASTEP TiO<sub>2</sub> rutile structural file [rutile-out.cell](../assets/rutile-out.cell) produces the following output: 

> Wyckoff points symmetry report for rutile-out.cell  
> Space Group International Symbol: P4_2/mnm  
> Space Group Hall Number: 419  

| Absolute | Fractional | Hessian constraints | Occupied |
|---|---|---|---|
| `[0.000 0.000 0.000]` | `[0.000 0.000 0.000]` | none | X |
| `[0.000 0.000 1.470]` | `[0.000 0.000 0.500]` | none |  |
| `[0.000 2.336 0.000]` | `[0.000 0.500 0.000]` | none |  |
| `[0.000 2.336 0.735]` | `[0.000 0.500 0.250]` | none |  |
| **`[0.000 2.336 1.470]`** | **`[0.000 0.500 0.500]`** | **none** |  |
| `[0.000 2.336 2.205]` | `[0.000 0.500 0.750]` | none |  |
| `[2.336 0.000 0.000]` | `[0.500 0.000 0.000]` | none |  |
| `[2.336 0.000 0.735]` | `[0.500 0.000 0.250]` | none |  |
| `[2.336 0.000 1.470]` | `[0.500 0.000 0.500]` | none |  |
| `[2.336 0.000 2.205]` | `[0.500 0.000 0.750]` | none |  |
| `[2.336 2.336 0.000]` | `[0.500 0.500 0.000]` | none |  |
| `[2.336 2.336 1.470]` | `[0.500 0.500 0.500]` | none | X |


An `X` in the `Occupied` column means that an atom of the host structure already sits on that Wyckoff point, so it is not available to the muon; the remaining points are the candidate sites. The coordinates in bold indicate the muon stopping site that agrees with the experimental results. The predicted stopping site is shown in the figure below.  As we mentioned above, the  muon is bonded to one of the six oxygen atoms that form an octahedron around the Ti<sup>3+</sup> atom, which is shown at the center of the TiO<sub>2</sub> rutile unit cell in this figure:

<img src="../../assets/rutile_muon.jpg" width="250" height="250" />

So, it is clear that there is a connection between the symmetry properties of the crystalline material and the potential muon stopping sites in that material.  The symmetry method is an extremely fast method.  However, if possible, this method should usually be used in conjunction with experimental results and physical intuition.  It is the case that there may be a significant number of high symmetry points in a given crystalline material and, unless there is some extra information available, it is difficult to discriminate between potential stopping sites. 


