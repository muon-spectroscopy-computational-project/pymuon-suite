# pymuon-suite
Collection of scripts and utilities for muon spectroscopy.

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

## Docker Environments

To use the Docker environments, you must have [Docker Desktop](https://www.docker.com/products/docker-desktop)
and [Docker Compose](https://docs.docker.com/compose/install/) installed.

Two images are available. The 'user' image provides a Python environment with pymuon-suite installed.
The 'dev' image, which must be used with its corresponding Docker Compose file, allows editing of the
pymuon-suite code.

### User

First, clone the repository:
```
git clone https://github.com/muon-spectroscopy-computational-project/pymuon-suite.git
```

Navigate to the `pymuon-suite` directory and run the following line to build the Docker image:
```
docker build -f Dockerfile_user -t pymuon-suite/user .
```

Once complete, you can run the container in interactive mode with
```
docker run -it pymuon-suite/user
```
or detached mode with
```
docker run -td pymuon-suite/user
```
or use the Docker Compose file [docker-compose-user.yaml](docker-compose-user.yaml),
which you can edit to mount a local folder into the container:
```
docker-compose -f docker-compose-user.yaml up -d
```

### Developer

First, clone the repository:
```
git clone https://github.com/muon-spectroscopy-computational-project/pymuon-suite.git
```

Navigate to the `pymuon-suite` directory and run the following line to build the Docker image:
```
docker build -f Dockerfile_dev -t pymuon-suite/dev .
```

Once complete, use the Docker Compose file [docker-compose-dev.yaml](docker-compose-dev.yaml) to
launch the container. This will mount your local files onto the container and install the package
in editable mode. This means that any changes to files on the container will not only change the
behaviour of the pymuon-suite installation on the container, but also be reflected in your own
local files.
```
docker-compose -f docker-compose-dev.yaml up -d
```
