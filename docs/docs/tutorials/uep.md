## The Unperturbed Electrostatic Potential method

Here we present our software implementation of the Unperturbed Electrostatic Potential (UEP) Method: a method that requires only one DFT calculation, which is used to compute the electronic density of the host material. This, in turn, is used to calculate the minima of the host material’s electrostatic potential and to estimate the stopping site of the **diamagnetic muon**, relying on the approximation that the muon’s presence does not significantly affect its surroundings. 

The UEP method can be used with a complementary tool, which evaluates the interstitial high-symmetry sites in the host material that are not already occupied by an atom. In practice, very often, the muon stopping site will be a high-symmetry site of the crystal that has not already been occupied by another atom. Hence, the complementary tool performs a crystallographic analysis of the pure system as a useful first step. We have explained how this complementary tool works 
[here](./symmetry.md).

A detailed description of this methodology is given in this [paper by S. Sturniolo and L. Liborio](https://aip.scitation.org/doi/10.1063/5.0012381). 

### Details of the software implementation

The commands for running the UEP method are the following:

`pm-uep-opt` performs an optimization of the muon position by minimising the classical electrostatic forces 
— which originate from the unperturbed electrostatic potential — that act on the muon;
 	
`pm-uep-plot` produces 1D and 2D plots describing the UEP along specific paths or planes defined in the crystalline structure; and
	
`pm-muairss` its application is twofold: first, it is used to produce batches of structure files with a muon defect added following
a Poisson random distribution. These muonated structural files are the ones optimized with `pm-uep-opt` and, then, `pm-muairss`
is used again to analyse and cluster the results of these optimizations.
 
For the UEP scripts, parameter files in the YAML format are necessary. These scripts are run with the commands: 

`pm-uep-plot` `<parameters file>` 

`pm-uep-opt` `<parameters file>`
	
Finally, pm-muairss has both a “write” and a “read” mode: one to create the batches of structure files with a muon defect, and the other to interpret the results of their optimization. It also needs both a structure and a parameter YAML file as inputs. To avoid mistakenly overwriting important data, the default mode is read. The two modes are used as follows:

`pm-muairss` `-t` `w` `<structure file>` `<parameters file>` # To write 

`pm-muairss` `-t` `r` `<structure file>` `<parameters file>` # To read

The `-t` `r` argument is optional; if omitted, read mode is assumed. For the parameter files, each script has its own arguments that can be set using them. A detailed description of the specific accepted parameters for the YAML files, and their meaning for each script, is provided on the [`pm-uep-opt` / `pm-uep-plot`](../uep.md) and [`pm-muairss`](../muairss.md) reference pages.

An example of a possible work pipeline to use the UEP method could then be as follows:

* run a CASTEP calculation on a bulk structure (non muonated) to compute the electronic density to use as input for the UEP method;

* generate a number of muonated structures with a random distribution of starting positions with `pm-muairss` in the write mode;

* relax each of these structures with `pm-uep-opt`; and

* perform a cluster analysis on the relaxed structures to identify the muon stopping sites with `pm-muairss` in the read
mode.

The clusters with the largest number of muons will then roughly correspond to the minima of the electrostatic potential, 
which have the largest attraction basin and are likely to be representing a muon stopping site in that host material.

### Examples of application of the UEP method

#### Magnetite (Fe<sub>3</sub>O<sub>4</sub>)

Potential muon stopping sites in Fe<sub>3</sub>O<sub>4</sub> magnetite are  experimentally well known.  The stopping sites are:

* located within a planar region that is perpendicular to the (111) direction in the unit cell and
* situated within ≈1.5 Å of one of the oxygen atoms defining the planar region. 

The figure below shows the O(111) site predicted by the UEP method: 

<img src="../../assets/fe3o4_muon_1.jpg" width="250" height="250" />

And this other figure below shows an example of a planar region, perpendicular to the (111) direction, which is indicated in yellow: 

<img src="../../assets/fe3o4_muon_2-plane.jpg" width="250" height="250" />

The muon is located at ≈1.3 Å from its closest oxygen atom in this planar region.  A detailed description of the input and output files, resulting 
from the application of the UEP method to Fe<sub>3</sub>O<sub>4</sub>, is in this [document](../assets/supplement.pdf). 
