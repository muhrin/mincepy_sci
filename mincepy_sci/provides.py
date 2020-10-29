# -*- coding: utf-8 -*-
from . import pandas_types


def get_types():
    """The central entry point """
    types = list()
    types.extend(pandas_types.TYPES)

    return types
