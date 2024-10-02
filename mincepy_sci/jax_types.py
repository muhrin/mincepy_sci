from typing import Any
import uuid

import jax
import jax.numpy as jnp
import mincepy


class JaxArrayHelper(mincepy.TypeHelper):
    TYPE = jax.Array
    TYPE_ID = uuid.UUID("d7270e15-c6fb-4621-abd8-c47b3be8b839")

    def yield_hashables(self, array: jax.Array, hasher):  # pylint: disable=arguments-differ
        yield from hasher.yield_hashables(array.tolist())

    def eq(self, one, other) -> bool:
        return (one == other).all()

    def save_instance_state(
        self, array: jax.Array, _referencer
    ):  # pylint: disable=arguments-differ
        return {"array": array.tolist(), "dtype": str(array.dtype)}

    def new(self, encoded_saved_state: dict[str, Any]):
        return jnp.asarray(encoded_saved_state["array"], dtype=encoded_saved_state["dtype"])

    def load_instance_state(
        self, array: jax.Array, saved_state, _referencer
    ):  # pylint: disable=arguments-differ
        pass  # Nothing to do, did it all in new


TYPES = (JaxArrayHelper,)
