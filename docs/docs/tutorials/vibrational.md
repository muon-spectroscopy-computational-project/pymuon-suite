## Vibrational averaging

### What is vibrational averaging?

 Usually, the static lattice approximation is used in calculating properties of a system (e.g. in the ab initio DFT code CASTEP). This approximation assumes that all nuclei are static, classical objects whilst electrons have their quantum nature treated in full. However, for light nuclei in atoms such as muonium, quantum effects are highly important and neglecting them can produce inaccurate results. The term vibrational averaging refers to a set of methods used to approximate quantum effects on nuclei by "vibrating" the nuclei and averaging properties along those vibrations.

These methods typically involve calculating the vibrational modes of a molecule and displacing the atoms along those modes. A grid of displacements is thus created for each atom. The desired property is then sampled at each point on the grid and averaged using some weighting. See this [paper by B. Monserrat](https://journals.aps.org/prb/abstract/10.1103/PhysRevB.93.014302) for more information.

### The `pymuon-suite` implementation

`pm-nq`, the section of `pymuonsuite` relating to nuclear quantum effects, implements a general, high-level vibrational averaging function that can be extended to multiple methods and physical properties. The general workflow is as follows:

1. Calculate the phonon modes of a molecule or crystal containing a muon using CASTEP or DFTB+
2. Use `pm-nq` to write out a set of structure files in which the muon is displaced from its equilibrium position along the phonon mode eigenvectors
3. Calculate the value of the desired property at each displacement (also referred to as a grid point)
4. Use `pm-nq` to calculate an average for the property over all displacements

The following are the methods that are currently implemented and ready for use:

#### Independent sampling

This method treats the muonium as an isolated particle in a 3-D harmonic oscillator. The muonium is displaced independently in `grid_n` increments along each of its 3 most significant phonon modes. Other atoms in the system are not displaced. The muonium can be treated like this due to its low mass relative to other chemical species, which means that its vibrations are effectively decoupled from the rest of the system.

The average over the displacements will be weighted by the wavefunction of the harmonic oscillator.

#### Monte Carlo sampling

Monte Carlo integration can be used to give the expectation value for a desired observable. Here we sample `grid_n` points distributed according to the nuclear density at a given temperature. This in turn is given by a product of Gaussians, one for each mode in the system, whose width is temperature dependent. 

Increasing system size should not increase the number of sampling points required for a constant level of accuracy. This typically makes Monte Carlo integration cheaper than alternatives once the system becomes larger than a few hundred atoms.

### Using `pm-nq`

If `pymuonsuite` has been installed (see [Installation](../installation.md)), then `pm-nq` can be called from the command line. The write mode will require a YAML-format parameter file for `pm-nq`, the format and options of which are described in the parameter file section below, and also a file representing the structure of interest in a format readable by ASE. To call the write mode enter:

```
pm-nq "<structure>" "<parameter_file>" -t w
```

This will write out a set of files with the muon displaced according to one of the methods described above. Each displacement will be located in a directory with the format `<structure>_displaced_<n>` (but with the file extension of `<structure>` removed). 

If the independent sampling method is used, then `grid_n*3` files will be written, with the first third being displacements along the muon's first major phonon mode, the second third corresponding to the second mode and so forth. If the Monte Carlo method is used then `grid_n` files will be written.

Once the displaced files have been generated, you must calculate the desired quantity for each, keeping the resulting data files in the same directory structure. The read/average mode can then be called by changing or omitting the task flag `-t`:

```
pm-nq "<structure>" "<parameter_file>" -t r
```

This will perform the appropriate average over the data files and produce a `averages.dat` file with the averaged quantities, alongside other files depending on the method used and property averaged over.

#### The parameter file

In order to use the vibrational averaging function, a YAML format parameter file must be created. The YAML format in this case is simply a set of entries, each on its own line, given by a parameter-value pair which are separated by a colon. All parameters are optional, and will use a default value if not provided.

The full list of accepted keywords, along with their meaning and default values, is given on the [`pm-nq` reference page](../quantum.md).

**Example Parameter File**

```
phonon_source_file: ethyleneMu_opt.phonons.pkl
phonon_source_type: dftb+
calculator:         dftb+
avgprop:            charge
write_allconf:      true
dftb_pbc:           false
grid_n:             2
mu_index:           -1
```
