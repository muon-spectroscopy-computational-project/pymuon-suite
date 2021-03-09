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

## Docker Environments

You must have [Docker Desktop](https://www.docker.com/products/docker-desktop)
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
