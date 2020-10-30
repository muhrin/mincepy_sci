# -*- coding: utf-8 -*-
# pylint: disable=wrong-import-position, invalid-name
import pytest

numpy = pytest.importorskip('numpy')

import mincepy


def test_saving_numpy_arrays(historian: mincepy.Historian):
    array = numpy.ones(10)

    array_id = historian.save(array)
    del array
    loaded_array = historian.load(array_id)
    assert all(loaded_array == numpy.ones(10))

    loaded_array[0] = 5.
    historian.save(loaded_array)


def test_saving_numpy_arrays_integers(historian: mincepy.Historian):
    array = numpy.ones(10, dtype=numpy.int64)

    array_id = historian.save(array)
    del array
    loaded_array = historian.load(array_id)
    assert all(loaded_array == numpy.ones(10, dtype=numpy.int64))

    loaded_array[0] = 5.
    historian.save(loaded_array)
