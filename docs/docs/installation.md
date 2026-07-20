# Installation

Requires Python 3.10+. Install with pip or conda:

```
pip install pymuonsuite
```

```
conda install pymuonsuite
```

If you plan to read CASTEP `.phonon` files, you will also need to install
[Euphonic](https://github.com/pace-neutrons/Euphonic):

```
pip install euphonic
```

```
conda install euphonic
```

Help with Euphonic installation can be found in the
[Euphonic documentation](https://euphonic.readthedocs.io/en/latest/installation.html).

## `spglib` build failed during pip install

On some platforms, additional tools are needed to build the `spglib` Python
module when installing via pip. On Windows, you may need to install
[Microsoft C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/);
on Linux you may need to `apt-get install python-dev` or
`yum install python-devel` according to your distribution. This should not be
necessary if installing via conda, and so we recommend using conda if you want
to avoid installing these tools.

Further help with Spglib installation can be found in the
[Spglib documentation](https://spglib.readthedocs.io/en/latest/python-interface.html).

## Development installation

To set up a development environment (using [uv](https://docs.astral.sh/uv/),
conda or Docker), see
[`DEVELOPMENT.md`](https://github.com/muon-spectroscopy-computational-project/pymuon-suite/blob/main/DEVELOPMENT.md)
in the repository. The short version, with uv:

```bash
git clone https://github.com/muon-spectroscopy-computational-project/pymuon-suite.git
cd pymuon-suite
uv sync --all-extras --group dev --group lint
```

Run commands inside this environment with `uv run`, e.g.
`uv run pytest pymuonsuite/test/` or `uv run pm-muairss --help`.
