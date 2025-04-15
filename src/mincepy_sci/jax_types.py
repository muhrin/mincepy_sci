from typing import Any
import uuid

import jax
import jax.numpy as jnp
import mincepy
from typing_extensions import override


class JaxArrayHelper(
    mincepy.TypeHelper,
    obj_type=(jax.Array, jax._src.array.ArrayImpl),  # pylint: disable=protected-access
    type_id=uuid.UUID("d7270e15-c6fb-4621-abd8-c47b3be8b839"),
):
    @override
    def yield_hashables(self, array: jax.Array, hasher, /):
        yield from hasher.yield_hashables(array.tolist())

    @override
    def eq(self, one, other, /) -> bool:
        return (one == other).all()

    @override
    def save_instance_state(self, array: jax.Array, _referencer, /):
        return {"array": array.tolist(), "dtype": str(array.dtype)}

    def new(self, encoded_saved_state: dict[str, Any], /):
        return jnp.asarray(encoded_saved_state["array"], dtype=encoded_saved_state["dtype"])

    def load_instance_state(self, array: jax.Array, saved_state, _referencer, /):
        pass  # Nothing to do, did it all in new


TYPES = (JaxArrayHelper,)
