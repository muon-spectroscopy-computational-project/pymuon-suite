import pathlib
from setuptools import setup, find_packages

from pymuonsuite import __version__

this_dir = pathlib.Path(__file__).parent
readme_text = (this_dir / "README.md").read_text()

if __name__ == "__main__":

    # Scan for Slater-Koster files data tree
    import os
    import glob

    sk_pkgdata = []
    fdir = os.path.dirname(__file__)

    for g in glob.glob(os.path.join(fdir, "pymuonsuite/data/dftb_pars/*")):
        if not os.path.isdir(g):
            continue
        setname = os.path.split(g)[1]
        sk_pkgdata += [
            os.path.join("data/dftb_pars/", setname, "*"),
            os.path.join("data/dftb_pars/", setname, setname, "*.skf"),
            os.path.join("data/dftb_pars/", "*"),
        ]

    setup(
        name="PyMuonSuite",
        version=__version__,
        description=("A suite of utilities for muon spectroscopy"),
        long_description=readme_text,
        long_description_content_type="text/markdown",
        author="Simone Sturniolo",
        author_email="simone.sturniolo@stfc.ac.uk",
        license="GPLv3",
        classifiers=[
            # How mature is this project? Common values are
            #   3 - Alpha
            #   4 - Beta
            #   5 - Production/Stable
            "Development Status :: 4 - Beta",
            # Indicate who your project is intended for
            "Intended Audience :: Science/Research",
            "Topic :: Scientific/Engineering :: Chemistry",
            "Topic :: Scientific/Engineering :: Physics",
            "Topic :: Scientific/Engineering :: Information Analysis",
            # Pick your license as you wish (should match "license" above)
            "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
            # Specify the Python versions you support here. In particular,
            # ensure that you indicate whether you support Python 2, Python 3
            # or both.
            "Programming Language :: Python :: 3",
        ],
        packages=find_packages(),
        # Requirements
        install_requires=[
            "numpy",
            "scipy",
            "ase>=3.18.1",
            "pyyaml",
            "schema",
            "spglib>0.8",
            "soprano>=0.8.11",
            "parse-fmt>=0.5",
        ],
        extras_require={
            "dev": ["pytest", "flake8", "black>=22.3.0", "pre-commit"],
            "phonons": ["euphonic"],
        },
        package_data={"pymuonsuite": sk_pkgdata},
        entry_points={
            "console_scripts": [
                ("pm-muairss-gen = " "pymuonsuite.muairss:main_generate"),
                ("pm-muairss = " "pymuonsuite.muairss:main"),
                ("pm-uep-opt = " "pymuonsuite.calculate.uep.__main__:geomopt_entry"),
                ("pm-uep-plot = " "pymuonsuite.calculate.uep.__main__:plot_entry"),
                ("pm-nq = " "pymuonsuite.quantum.__main__:nq_entry"),
                ("pm-asephonons = " "pymuonsuite.quantum.__main__:asephonons_entry"),
                ("pm-symmetry = " "pymuonsuite.symmetry:main"),
            ]
        },
        python_requires=">=3.7, <=3.11",
    )
