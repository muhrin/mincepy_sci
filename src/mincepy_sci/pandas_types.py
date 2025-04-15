import uuid

import mincepy
import pandas as pd
from typing_extensions import override


class DataFrameHelper(
    mincepy.TypeHelper,
    obj_type=pd.DataFrame,
    type_id=uuid.UUID("0fc8dc7b-7378-4e5a-ba1f-ad8dcb0dd3c8"),
):

    @override
    def eq(self, one: pd.DataFrame, other: pd.DataFrame, /) -> bool:
        if not isinstance(one, pd.DataFrame) or not isinstance(other, pd.DataFrame):
            return False

        return one.equals(other)

    @override
    def yield_hashables(self, obj, hasher, /):
        yield from hasher.yield_hashables(obj.values.tobytes())

    @override
    def save_instance_state(self, obj: pd.DataFrame, saver, /):
        return obj.to_dict(orient="split")

    @override
    def load_instance_state(self, obj, saved_state, loader: "mincepy.Loader", /):
        """Take the given blank object and load the instance state into it"""
        obj.__init__(**saved_state)  # pylint: disable=unnecessary-dunder-call


TYPES = (DataFrameHelper,)
