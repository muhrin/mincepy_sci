# -*- coding: utf-8 -*-
import uuid

import mincepy
import pandas


class DataFrameHelper(mincepy.TypeHelper):
    TYPE = pandas.DataFrame
    TYPE_ID = uuid.UUID("0fc8dc7b-7378-4e5a-ba1f-ad8dcb0dd3c8")

    def eq(self, one: pandas.DataFrame, other: pandas.DataFrame) -> bool:
        if not isinstance(one, pandas.DataFrame) or not isinstance(
            other, pandas.DataFrame
        ):
            return False

        return one.equals(other)

    def yield_hashables(self, obj, hasher):
        yield from hasher.yield_hashables(obj.values.tobytes())

    def save_instance_state(
        self, obj: pandas.DataFrame, saver
    ):  # pylint: disable=unused-argument
        return obj.to_dict(orient="split")

    def load_instance_state(
        self, obj, saved_state, loader: "mincepy.Loader"
    ):  # pylint: disable=unused-argument
        """Take the given blank object and load the instance state into it"""
        obj.__init__(**saved_state)


TYPES = (DataFrameHelper(),)
