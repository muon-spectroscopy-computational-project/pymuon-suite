# Python 2-to-3 compatibility code
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from setuptools import setup, find_packages

if __name__ == '__main__':

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
          entry_points={
              'console_scripts': [
                  ('pm-muairss-gen = '
                   'pymuonsuite.muairss:main'),
                  ('pm-uep-opt = '
                   'pymuonsuite.calculate.uep.__main__:geomopt_entry')
              ]
          },
          python_requires='>=2.7'
          )
