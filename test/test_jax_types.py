# pylint: disable=wrong-import-position, invalid-name
import jax.numpy as jnp
import mincepy
import pytest

numpy = pytest.importorskip("numpy")


def test_saving_jax_arrays(historian: mincepy.Historian):
    array = jnp.ones(10)

    array_id = historian.save(array)
    del array
    loaded_array = historian.load(array_id)
    assert all(loaded_array == jnp.ones(10))

    loaded_array[0] = 5.0
    historian.save(loaded_array)


def test_saving_jax_arrays_integers(historian: mincepy.Historian):
    array = jnp.ones(10, dtype=jnp.int64)

    array_id = historian.save(array)
    del array
    loaded_array = historian.load(array_id)
    assert all(loaded_array == jnp.ones(10, dtype=jnp.int64))

    loaded_array[0] = 5.0
    historian.save(loaded_array)
