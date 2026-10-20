import uuid

import mincepy
import numpy as np
from typing_extensions import override

from . import arrays


class ArrayHelper(
    mincepy.TypeHelper,
    obj_type=np.ndarray,
    type_id=uuid.UUID("eff7de75-2d6c-48dd-9b46-0ce16fb8b688"),
):
    """Arrays are stored as bytes, keeping their exact dtype and shape (see :mod:`.arrays`)"""

    @override
    def yield_hashables(self, array: np.ndarray, hasher, /):
        yield from arrays.yield_hashables(array, hasher)

    @override
    def eq(self, one, other, /) -> bool:
        return arrays.arrays_equal(one, other)

    @override
    def save_instance_state(self, array: np.ndarray, saver, /):
        return arrays.save_state(array, saver)

    @override
    def new(self, encoded_saved_state, /):
        if arrays.is_stored_array(encoded_saved_state):
            # The contents are filled in by load_instance_state()
            return arrays.empty_like_saved(encoded_saved_state)

        # Legacy format: the array was saved as (nested) lists, without its dtype
        return np.asarray(encoded_saved_state)

    @override
    def load_instance_state(self, array: np.ndarray, saved_state, _referencer, /):
        if arrays.is_stored_array(saved_state):
            array[...] = arrays.load_array(saved_state)


TYPES = (ArrayHelper,)
