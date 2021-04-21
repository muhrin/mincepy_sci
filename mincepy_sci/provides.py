# -*- coding: utf-8 -*-
from . import ase_types
from . import numpy_types
from . import pandas_types
from . import plams_types
from . import pymatgen_types


def get_types():
    """The central entry point """
    types = list()
    types.extend(ase_types.TYPES)
    types.extend(numpy_types.TYPES)
    types.extend(pandas_types.TYPES)
    types.extend(plams_types.TYPES)
    types.extend(pymatgen_types.TYPES)

    return types
