from typing import Any
import uuid

import jax
import jax.numpy as jnp
import mincepy
import numpy as np
from typing_extensions import override

from . import arrays


class JaxArrayHelper(
    mincepy.TypeHelper,
    obj_type=(jax.Array, jax._src.array.ArrayImpl),
    type_id=uuid.UUID("d7270e15-c6fb-4621-abd8-c47b3be8b839"),
    # JAX arrays can't be changed, so they have to be created complete in new(), for which mincePy
    # needs to decode their saved state (including any file) first
    immutable=True,
):
    """Arrays are stored as bytes, keeping their exact dtype and shape (see :mod:`.arrays`)"""

    @override
    def yield_hashables(self, array: jax.Array, hasher, /):
        yield from arrays.yield_hashables(np.asarray(array), hasher)

    @override
    def eq(self, one, other, /) -> bool:
        return arrays.arrays_equal(np.asarray(one), np.asarray(other))

    @override
    def save_instance_state(self, array: jax.Array, saver, /):
        return arrays.save_state(np.asarray(array), saver)

    @override
    def new(self, saved_state: dict[str, Any], /):
        if arrays.is_stored_array(saved_state):
            return jnp.asarray(arrays.load_array(saved_state))

        # Legacy format: the array was saved as (nested) lists
        return jnp.asarray(saved_state["array"], dtype=saved_state["dtype"])

    @override
    def load_instance_state(self, array: jax.Array, saved_state, _referencer, /):
        pass  # Nothing to do, it's all done in new()


TYPES = (JaxArrayHelper,)
