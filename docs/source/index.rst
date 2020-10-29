.. mincepy documentation master file, created by
   sphinx-quickstart on Fri Mar 31 17:03:20 2017.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. _mincePy: https://github.com/muhrin/mincepy


Welcome to mincePy Sci's documentation!
=======================================

.. image:: https://codecov.io/gh/muhrin/mincepy/branch/develop/graph/badge.svg
    :target: https://codecov.io/gh/muhrin/mincepy_sci
    :alt: Coveralls

.. image:: https://travis-ci.org/muhrin/mincepy.svg
    :target: https://travis-ci.org/muhrin/mincepy_sci
    :alt: Travis CI

.. image:: https://img.shields.io/pypi/v/mincepy.svg
    :target: https://pypi.python.org/pypi/mincepy_sci/
    :alt: Latest Version

.. image:: https://img.shields.io/pypi/wheel/mincepy.svg
    :target: https://pypi.python.org/pypi/mincepy_sci/

.. image:: https://img.shields.io/pypi/pyversions/mincepy.svg
    :target: https://pypi.python.org/pypi/mincepy_sci/

.. image:: https://img.shields.io/pypi/l/mincepy.svg
    :target: https://pypi.python.org/pypi/mincepy_sci/


`mincePy`_: move the database to one side and let your objects take centre stage.

MincePy Sci is a set of plugins for `mincePy`_ that enable common scientific data formats to be saved.


Installation
++++++++++++

Installation with pip:

.. code-block:: shell

    pip install mincepy_sci


Installation from git:

.. code-block:: shell

    # via pip
    pip install https://github.com/muhrin/mincepy_sci/archive/master.zip

    # manually
    git clone https://github.com/muhrin/mincepy_sci.git
    cd mincepy
    python setup.py install


Next you'll need MongoDB, in Ubuntu it's as simple as:


.. code-block:: shell

    apt install mongodb

see `here <https://docs.mongodb.com/manual/administration/install-community/>`_, for other platforms.


Table Of Contents
+++++++++++++++++

.. toctree::
   :glob:
   :maxdepth: 3

   development
   apidoc


Versioning
++++++++++

This software follows `Semantic Versioning`_


.. _Semantic Versioning: http://semver.org/
