# -*- coding: utf-8 -*-
import importlib


def get_types():
    """The central entry point"""
    types = list()
    _extend(types, "ase_types")
    _extend(types, "e3nn_types")
    _extend(types, "numpy_types")
    _extend(types, "pandas_types")
    _extend(types, "plams_types")
    _extend(types, "pyilt2_types")
    _extend(types, "pymatgen_types")
    _extend(types, "pytorch_types")
    _extend(types, "rdkit_types")

    return types


def _extend(types: list, module_name: str):
    try:
        mod = importlib.import_module(f"mincepy_sci.{module_name}")
    except ImportError:
        pass
    else:
        types.extend(mod.TYPES)
