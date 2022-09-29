.. _mincePy: https://github.com/muhrin/mincepy
.. _mincepy documentation: https://mincepy.readthedocs.org/
.. _issue: https://github.com/muhrin/mincepy_sci/issues

.. _ase: https://wiki.fysik.dtu.dk/ase/
.. _numpy: https://numpy.org/
.. _pandas: https://pandas.pydata.org/
.. _plams: https://www.scm.com/doc/plams/index.html
.. _pyilt2: http://wgserve.de/pyilt2/
.. _pymatgen: https://pymatgen.org/
.. _pytorch: https://pytorch.org/
.. _rdkit: https://www.rdkit.org/


mincePy Sci
===========

.. image:: https://codecov.io/gh/muhrin/mincepy_sci/branch/develop/graph/badge.svg
    :target: https://codecov.io/gh/muhrin/mincepy_sci
    :alt: Coverage

.. image:: https://travis-ci.com/muhrin/mincepy_sci.svg?branch=master
    :target: https://travis-ci.com/github/muhrin/mincepy_sci
    :alt: Travis CI

.. image:: https://img.shields.io/pypi/v/mincepy-sci.svg
    :target: https://pypi.python.org/pypi/mincepy_sci/
    :alt: Latest Version

.. image:: https://img.shields.io/pypi/pyversions/mincepy-sci.svg
    :target: https://pypi.python.org/pypi/mincepy_sci/

.. image:: https://img.shields.io/pypi/l/mincepy-sci.svg
    :target: https://pypi.python.org/pypi/mincepy_sci/


mincePy: move the database to one side and let your objects take centre stage.

MincePy Sci is a set of plugins for `mincePy`_ that enable common scientific data formats to be saved.

See `mincepy documentation`_ for more information.

This plugin provide support for:

`ase`_:
    * ``Atoms``
    * ``Cell``

`numpy`_:
    * ``ndarray``


`pandas`_:
    * ``DataFrame``


`plams`_:
    * ``Settings``
    * ``Molecule``
    * ``Results``


`pyilt2`_:
    * ``dataset``

(unfortunately the pypi version is out of date and so you should use an updated version from `here <https://github.com/muhrin/pyilt2>`_).

`pymatgen`_:
    * ``Structure``
    * ``Molecule``
    * ``BandStructure``
    * ``CompleteDos``
    * ``PeriodicSite``


`pytorch`_:
    * ``Tensor``
    * ``Module`` (using mixin or type helper)

`rdkit`_:
    * ``Mol``

Contributing
------------

We'd love to get more data types supported by `mincePy`_!
For now, the easiest way is to follow the examples set by the existing types and contribute a PR.
If you'd like more (i.e. any) documentation on how to do this or would like to request that someone with more expertise makes a plugin just create an `issue`_ and we'll see what we can do.
