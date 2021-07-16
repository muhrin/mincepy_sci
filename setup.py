# -*- coding: utf-8 -*-
from setuptools import setup

__author__ = 'Martin Uhrin'
__license__ = 'LGPLv3'

about = {}
with open('mincepy_sci/version.py') as f:
    exec(f.read(), about)

setup(name='mincepy_sci',
      version=about['__version__'],
      description=
      'Plugins to enable common scientific and machine learning type to be saved by mincePy',
      long_description=open('README.rst').read(),
      url='https://github.com/muhrin/mincepy_sci.git',
      author='Martin Uhrin',
      author_email='martin.uhrin.10@ucl.ac.uk',
      license=__license__,
      classifiers=[
          'License :: OSI Approved :: MIT License',
          'License :: OSI Approved :: GNU Lesser General Public License v3 or later (LGPLv3+)',
          'Programming Language :: Python :: 3.5',
          'Programming Language :: Python :: 3.6',
          'Programming Language :: Python :: 3.7',
          'Programming Language :: Python :: 3.8',
      ],
      keywords='database schemaless nosql orm object-store concurrent optimistic-locking',
      install_requires=['mincepy>=0.15.20'],
      extras_require={
          'dev': [
              'pip',
              'pytest>4',
              'pytest-cov',
              'pre-commit',
              'prospector',
              'pylint',
              'twine',
              'yapf',
          ],
          'full': [
              'ase',
              'numpy',
              'pandas',
              'plams',
              'pyilt22',
              'pymatgen',
          ],
      },
      packages=['mincepy_sci'],
      include_package_data=True,
      test_suite='test',
      entry_points={
          'mincepy.plugins.types': ['mincepy_sci = mincepy_sci.provides:get_types'],
      })
