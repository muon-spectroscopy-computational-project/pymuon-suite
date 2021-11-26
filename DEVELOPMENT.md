# Developing pymuonsuite

You will need to set up a development environment. **Using conda is recommended** (but not documented here).
Docker environments are also available and are described below.

## Pre-commit Hook

There is a pre-commit hook available which will automatically format your code to follow those style guidelines
which are enforced by the GitHub Actions workflow. It is not mandatory to use the hook, but it will save you time
fixing code style!

To use it, download the `pre-commit` package:

For pip:
```
pip install pre-commit
pre-commit install
```

For conda:
```
conda install -c conda-forge pre-commit
pre-commit install
```

The hook is configured in `.pre-commit-config.yaml`. `pre-commit install` automatically finds
this file and converts it to a script which is placed in `.git/hooks/pre-commit`.

Now when you run `git commit` you should see the hook run and any failures will be flagged.

More information at https://pre-commit.com/

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
