"""Storing arrays in mincePy.

Arrays are stored as their raw bytes together with their exact dtype and shape.  Small arrays are
kept in the record itself, larger ones (see :data:`INLINE_LIMIT`) are stored in a mincePy file so
that records stay small however much data an object holds (MongoDB limits documents to 16MB).
This is used by the numpy and JAX type helpers.
"""

from typing import Any, BinaryIO, cast

import mincepy
import numpy as np

__all__ = ("INLINE_LIMIT",)

#: Arrays larger than this (in bytes) are stored in a mincePy file rather than in the record
INLINE_LIMIT = 256 * 1024

FORMAT = "format"
VERSION = 1
DTYPE = "dtype"
SHAPE = "shape"
DATA = "data"
FILE = "file"
ITEMS = "items"
OBJECT = "object"


def is_stored_array(saved_state: Any) -> bool:
    """Is this a saved state in the current format (rather than a legacy one)?"""
    return isinstance(saved_state, dict) and saved_state.get(FORMAT) == VERSION


def save_state(array: np.ndarray, saver: "mincepy.Saver") -> dict:
    """Get the state to save for an array"""
    state: dict[str, Any] = {FORMAT: VERSION, SHAPE: list(array.shape)}
    if array.dtype.kind == "O":
        # Object arrays can't be stored as bytes, so save their elements (as mincePy objects)
        state[DTYPE] = OBJECT
        state[ITEMS] = array.ravel().tolist()
        return state

    state[DTYPE] = dtype_spec(array.dtype)
    data = np.ascontiguousarray(array).tobytes()
    if len(data) <= INLINE_LIMIT:
        state[DATA] = data
    else:
        array_file = saver.historian.create_file("array.bin")
        with cast(BinaryIO, array_file.open("wb")) as stream:
            stream.write(data)
        state[FILE] = array_file

    return state


def empty_like_saved(saved_state: dict) -> np.ndarray:
    """Create an (uninitialised) array with the dtype and shape of the saved state"""
    dtype = object if saved_state[DTYPE] == OBJECT else to_dtype(saved_state[DTYPE])
    return np.empty(tuple(saved_state[SHAPE]), dtype=dtype)


def load_array(saved_state: dict) -> np.ndarray:
    """Load an array from its (decoded) saved state"""
    shape = tuple(saved_state[SHAPE])
    if saved_state[DTYPE] == OBJECT:
        items = saved_state[ITEMS]
        array = np.empty(len(items), dtype=object)
        array[:] = items
        return array.reshape(shape)

    if FILE in saved_state:
        with cast(BinaryIO, saved_state[FILE].open("rb")) as stream:
            data = stream.read()
    else:
        data = saved_state[DATA]

    return np.frombuffer(data, dtype=to_dtype(saved_state[DTYPE])).reshape(shape).copy()


def yield_hashables(array: np.ndarray, hasher):
    """Hash the dtype, shape and contents of an array"""
    if array.dtype.kind == "O":
        yield from hasher.yield_hashables([OBJECT, list(array.shape), array.ravel().tolist()])
        return

    yield from hasher.yield_hashables([dtype_spec(array.dtype), list(array.shape)])
    yield np.ascontiguousarray(array).tobytes()


def arrays_equal(one: np.ndarray, other: np.ndarray) -> bool:
    """Arrays are equal if they have the same dtype, shape and contents (with NaNs equal to each
    other, so that an unchanged array containing NaNs isn't seen as having changed)"""
    if one.dtype != other.dtype or one.shape != other.shape:
        return False

    equal_nan = one.dtype.kind in "fc"
    return bool(np.array_equal(one, other, equal_nan=equal_nan))


def dtype_spec(dtype: np.dtype) -> str:
    """Get a string from which the dtype can be recreated exactly (including byte order)"""
    if dtype.kind == "V" and dtype.names is None:
        # Extension types (e.g. bfloat16 from ml_dtypes) can only be identified by name
        return dtype.name
    if dtype.names is not None:
        raise TypeError(f"Structured arrays are not supported, got dtype '{dtype}'")

    return dtype.str


def to_dtype(spec: str) -> np.dtype:
    """Get the dtype from its spec (see :func:`dtype_spec`)"""
    try:
        return np.dtype(spec)
    except TypeError:
        # It may be one of the ml_dtypes types (e.g. bfloat16) that numpy only knows about once
        # ml_dtypes has been imported
        try:
            import ml_dtypes
        except ImportError:
            raise TypeError(
                f"Unknown dtype '{spec}' (it may need the ml_dtypes package installing)"
            ) from None
        return np.dtype(getattr(ml_dtypes, spec))
