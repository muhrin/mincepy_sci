"""Tests for how numpy and JAX arrays are stored: exactly (dtype and shape), compactly (as bytes)
and without hitting database document limits (large arrays go in files)"""

import mincepy
import pytest

np = pytest.importorskip("numpy")

from mincepy_sci import arrays, numpy_types  # noqa: E402

DTYPES = [
    "bool",
    "int8",
    "int16",
    "int32",
    "int64",
    "uint8",
    "uint64",
    "float16",
    "float32",
    "float64",
    "complex64",
    "complex128",
    ">f4",  # Non-native byte order
    "datetime64[ns]",
    "U5",
]


@pytest.fixture
def reload(archive_uri):
    """Load objects using a new connection, so that they really come from the database"""

    def load(obj_id):
        return mincepy.connect(archive_uri).load(obj_id)

    return load


@pytest.fixture
def round_trip(historian: mincepy.Historian, reload):
    def save_and_load(obj):
        obj_id = historian.save(obj)
        return reload(obj_id), obj_id

    return save_and_load


@pytest.mark.parametrize("dtype", DTYPES)
def test_numpy_dtypes_are_preserved(round_trip, dtype):
    array = np.arange(6).astype(dtype).reshape(2, 3)
    loaded, _ = round_trip(array.copy())
    assert loaded.dtype == np.dtype(dtype)
    assert loaded.shape == (2, 3)
    assert np.array_equal(loaded, array)


@pytest.mark.parametrize("shape", [(), (0,), (3, 0), (2, 3, 4)])
def test_numpy_shapes_are_preserved(round_trip, shape):
    array = np.ones(shape, dtype=np.float32)
    loaded, _ = round_trip(array)
    assert loaded.shape == shape
    assert loaded.dtype == np.float32


def test_numpy_fortran_order(round_trip):
    array = np.asfortranarray(np.arange(12.0).reshape(3, 4))
    loaded, _ = round_trip(array)
    assert np.array_equal(loaded, np.arange(12.0).reshape(3, 4))


def test_numpy_loaded_arrays_are_writable(archive_uri, round_trip, reload):
    _loaded, obj_id = round_trip(np.zeros(4))
    other = mincepy.connect(archive_uri)
    loaded = other.load(obj_id)
    loaded[0] = 5.0
    other.save(loaded)
    assert reload(obj_id)[0] == 5.0


def test_numpy_object_arrays(round_trip):
    array = np.array(["a", 1, None], dtype=object)
    loaded, _ = round_trip(array)
    assert loaded.dtype == object
    assert list(loaded) == ["a", 1, None]


def test_small_arrays_are_inline_large_ones_in_files(
    historian: mincepy.Historian, reload, monkeypatch
):
    monkeypatch.setattr(arrays, "INLINE_LIMIT", 1024)
    small_id = historian.save(np.zeros(10))
    large_id = historian.save(np.arange(1000.0))  # 8000 bytes

    small_state = historian.records.get(small_id).state
    large_state = historian.records.get(large_id).state
    assert arrays.DATA in small_state and arrays.FILE not in small_state
    assert arrays.FILE in large_state and arrays.DATA not in large_state

    assert np.array_equal(reload(large_id), np.arange(1000.0))


def test_large_arrays(historian: mincepy.Historian, round_trip):
    """Larger than a MongoDB document can be (16MB)"""
    array = np.random.default_rng(0).random(3_000_000)  # 24MB
    loaded, obj_id = round_trip(array.copy())
    assert np.array_equal(loaded, array)
    assert arrays.FILE in historian.records.get(obj_id).state


def test_nan_arrays_are_unchanged_when_resaved(historian: mincepy.Historian):
    array = np.array([1.0, np.nan])
    obj_id = historian.save(array)
    historian.save(array)
    assert len(historian.history(obj_id)) == 1


def test_legacy_numpy_format(historian: mincepy.Historian, reload, monkeypatch):
    """Arrays saved by older versions (as nested lists) should still load"""
    monkeypatch.setattr(
        numpy_types.ArrayHelper,
        "save_instance_state",
        lambda self, array, _saver, /: array.tolist(),
    )
    obj_id = historian.save(np.arange(6.0).reshape(2, 3))
    monkeypatch.undo()

    assert np.array_equal(reload(obj_id), np.arange(6.0).reshape(2, 3))


def test_bfloat16(round_trip):
    ml_dtypes = pytest.importorskip("ml_dtypes")
    array = np.arange(4).astype(ml_dtypes.bfloat16)
    loaded, _ = round_trip(array)
    assert loaded.dtype == ml_dtypes.bfloat16
    assert np.array_equal(loaded.astype(np.float32), np.arange(4.0, dtype=np.float32))


# region JAX


@pytest.fixture
def jnp():
    return pytest.importorskip("jax.numpy")


@pytest.mark.parametrize("dtype", ["float32", "int32", "bool", "bfloat16", "float16", "uint8"])
def test_jax_dtypes_are_preserved(round_trip, jnp, dtype):
    array = jnp.arange(6).astype(dtype).reshape(2, 3)
    loaded, _ = round_trip(array)
    assert loaded.dtype == array.dtype
    assert loaded.shape == (2, 3)
    assert jnp.array_equal(loaded, array)


def test_jax_large_arrays(historian: mincepy.Historian, round_trip, jnp, monkeypatch):
    monkeypatch.setattr(arrays, "INLINE_LIMIT", 1024)
    array = jnp.arange(1000.0)
    loaded, obj_id = round_trip(array)
    assert jnp.array_equal(loaded, array)
    assert arrays.FILE in historian.records.get(obj_id).state


def test_legacy_jax_format(historian: mincepy.Historian, reload, jnp, monkeypatch):
    from mincepy_sci import jax_types

    monkeypatch.setattr(
        jax_types.JaxArrayHelper,
        "save_instance_state",
        lambda self, array, _saver, /: {"array": array.tolist(), "dtype": str(array.dtype)},
    )
    obj_id = historian.save(jnp.arange(4, dtype=jnp.int32))
    monkeypatch.undo()

    loaded = reload(obj_id)
    assert loaded.dtype == jnp.int32
    assert jnp.array_equal(loaded, jnp.arange(4, dtype=jnp.int32))


# endregion
