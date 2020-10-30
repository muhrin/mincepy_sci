# -*- coding: utf-8 -*-
try:
    import pandas
except ImportError:
    TYPES = tuple()
else:
    import uuid

    import mincepy

    class DataFrameHelper(mincepy.BaseHelper):
        TYPE = pandas.DataFrame
        TYPE_ID = uuid.UUID('0fc8dc7b-7378-4e5a-ba1f-ad8dcb0dd3c8')

        def yield_hashables(self, obj, hasher):
            yield from hasher.yield_hashables(obj.to_dict())

        def save_instance_state(self, obj: pandas.DataFrame, saver):  # pylint: disable=unused-argument
            return obj.to_dict(orient='split')

        def load_instance_state(self, obj, saved_state, loader: 'mincepy.Loader'):  # pylint: disable=unused-argument
            """Take the given blank object and load the instance state into it"""
            obj.__init__(**saved_state)

    TYPES = (DataFrameHelper(),)
