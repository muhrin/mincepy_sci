import uuid

import mincepy
import numpy as np
from typing_extensions import override


class ArrayHelper(
    mincepy.TypeHelper,
    obj_type=np.ndarray,
    type_id=uuid.UUID("eff7de75-2d6c-48dd-9b46-0ce16fb8b688"),
):

    @override
    def yield_hashables(self, array: np.ndarray, hasher, /):
        yield from hasher.yield_hashables(array.tolist())

    @override
    def eq(self, one, other, /) -> bool:
        return (one == other).all()

    @override
    def save_instance_state(self, array: np.ndarray, _referencer, /):
        return array.tolist()

    @override
    def new(self, encoded_saved_state, /):
        return np.asarray(encoded_saved_state)

    @override
    def load_instance_state(self, array: np.ndarray, saved_state, _referencer, /):
        pass  # Nothing to do, did it all in new


TYPES = (ArrayHelper,)
