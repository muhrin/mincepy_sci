.. _mincePy: https://github.com/muhrin/mincepy
.. _mincepy documentation: https://mincepy.readthedocs.io/
.. _issue: https://github.com/muhrin/mincepy_sci/issues

.. _ase: https://wiki.fysik.dtu.dk/ase/
.. _e3nn: https://docs.e3nn.org/en/stable/
.. _jax: https://jax.readthedocs.io/
.. _numpy: https://numpy.org/
.. _pandas: https://pandas.pydata.org/
.. _plams: https://www.scm.com/doc/plams/index.html
.. _pyilt2: https://pypi.org/project/pyilt22/
.. _pymatgen: https://pymatgen.org/
.. _pytorch: https://pytorch.org/
.. _rdkit: https://www.rdkit.org/


mincePy Sci
===========

.. image:: https://github.com/muhrin/mincepy_sci/actions/workflows/ci.yml/badge.svg
    :target: https://github.com/muhrin/mincepy_sci/actions/workflows/ci.yml
    :alt: CI

.. image:: https://codecov.io/gh/muhrin/mincepy_sci/branch/develop/graph/badge.svg
    :target: https://codecov.io/gh/muhrin/mincepy_sci
    :alt: Coverage

.. image:: https://img.shields.io/pypi/v/mincepy-sci.svg
    :target: https://pypi.python.org/pypi/mincepy-sci/
    :alt: Latest Version

.. image:: https://img.shields.io/python/required-version-toml?tomlFilePath=https%3A%2F%2Fraw.githubusercontent.com%2Fmuhrin%2Fmincepy_sci%2Fdevelop%2Fpyproject.toml
    :target: https://pypi.python.org/pypi/mincepy-sci/
    :alt: Python versions

.. image:: https://img.shields.io/pypi/l/mincepy-sci.svg
    :target: https://pypi.python.org/pypi/mincepy-sci/
    :alt: License

.. image:: https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json
    :target: https://github.com/astral-sh/ruff
    :alt: Ruff

mincePy: move the database to one side and let your objects take centre stage.

mincePy Sci is a set of plugins for `mincePy`_ that enable common scientific and machine learning
types to be saved.  See the `mincepy documentation`_ for more information.

Installation
------------

mincePy Sci needs Python 3.11 or later.  Install it with mincePy:

.. code-block:: shell

    pip install mincepy[sci]

or on its own, ``pip install mincepy-sci``.  Once installed, mincePy finds the plugins
automatically, and support for each library is enabled if that library is installed.  To install
all the supported libraries as well, use ``pip install mincepy-sci[full]``.

Supported types
---------------

`ase`_:
    * ``Atoms``
    * ``Cell``

`e3nn`_:
    * ``o3``:
        * ``Irrep`` and ``Irreps``
        * ``Instruction``
        * ``TensorProduct``
        * ``ElementwiseTensorProduct``
        * ``ReducedTensorProducts``
        * ``TensorSquare``
    * ``nn``:
        * ``Gate``
        * ``Extract``
        * ``Activation``
    * ``math``:
        * ``normalize2mom``

`jax`_:
    * ``Array``

`numpy`_:
    * ``ndarray``

`pandas`_:
    * ``DataFrame``

`plams`_:
    * ``Settings``
    * ``Molecule``
    * ``Atom``
    * ``Bond``

`pyilt2`_ (installed from PyPI as ``pyilt22``):
    * ``dataset``

`pymatgen`_:
    * ``Structure``
    * ``Molecule``
    * ``BandStructure``
    * ``CompleteDos``
    * ``PeriodicSite``

`pytorch`_:
    * ``Tensor``
    * ``Parameter``
    * ``Size``
    * ``Conv2d``
    * ``MaxPool2d``
    * ``Linear``
    * ``ModuleList``
    * ``ModuleDict``

`rdkit`_:
    * ``Mol``

Contributing
------------

We'd love to get more data types supported by `mincePy`_!
For now, the easiest way is to follow the examples set by the existing types and contribute a PR.
If you'd like more (i.e. any) documentation on how to do this or would like to request that someone
with more expertise makes a plugin just create an `issue`_ and we'll see what we can do.

To develop mincePy Sci (Python 3.11 or later):

.. code-block:: shell

    pip install -e '.[dev,full]'
    pre-commit install

The tests can be run against MongoDB (by default ``mongodb://127.0.0.1``) or, without a server,
using litemongo's in-memory engine:

.. code-block:: shell

    MINCEPY_TEST_URI="litemongo://?engine=memory#mincepy-sci-tests" pytest test

Types are checked with ty (``ty check``, also run in CI).  Files that don't pass yet are listed
under ``[tool.ty.src] exclude`` in ``pyproject.toml``, please remove them from there once fixed.
