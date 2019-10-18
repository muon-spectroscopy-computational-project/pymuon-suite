# Python 2-to-3 compatibility code
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from setuptools import setup, find_packages

if __name__ == '__main__':

    # Scan for Slater-Koster files data tree
    import os
    import glob
    sk_pkgdata = []
    fdir = os.path.dirname(__file__)

    for g in glob.glob(os.path.join(fdir, 'pymuonsuite/data/dftb_pars/*')):
        if not os.path.isdir(g):
            continue
        setname = os.path.split(g)[1]
        sk_pkgdata += [os.path.join('data/dftb_pars/', setname, '*'),
                       os.path.join('data/dftb_pars/', setname, setname,
                                    '*.skf')]

    setup(name='PyMuonSuite',
          version='0.0.1',
          description=('A suite of utilities for muon spectroscopy'),
          author='Simone Sturniolo',
          author_email='simone.sturniolo@stfc.ac.uk',
          packages=find_packages(),
          # Requirements
          install_requires=[
              'numpy',
              'scipy',
              'ase',
              'pyyaml',
              'schema',
              'spglib',
              'soprano',
              'parse-fmt>=0.5'
          ],
          package_data={'pymuonsuite': sk_pkgdata},
          entry_points={
              'console_scripts': [
                  ('pm-muairss-gen = '
                   'pymuonsuite.muairss:main_generate'),
                  ('pm-muairss = '
                   'pymuonsuite.muairss:main'),
                  ('pm-uep-opt = '
                   'pymuonsuite.calculate.uep.__main__:geomopt_entry'),
                  ('pm-nq = '
                   'pymuonsuite.quantum.__main__:nq_entry'),
                  ('pm-asephonons = '
                   'pymuonsuite.quantum.__main__:asephonons_entry')
              ]
          },
          python_requires='>=2.7'
          )
