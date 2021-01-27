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

## Installing with Docker

You must have [Docker Desktop](https://www.docker.com/products/docker-desktop)
and [Docker Compose](https://docs.docker.com/compose/install/) installed.

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

or use the Docker Compose file [docker-compose-user.yaml](docker-compose-user.yaml)
which you can edit to mount a local folder into the container:

```
docker-compose -f docker-compose-user.yaml up -d
```

### Developer

todo