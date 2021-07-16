# -*- coding: utf-8 -*-
try:
    import pyilt2
except ImportError:
    TYPES = tuple()
else:
    import uuid

    import mincepy

    class DatasetHelper(mincepy.TypeHelper):
        TYPE = pyilt2.dataset
        TYPE_ID = uuid.UUID('5d032ec2-31e3-41ae-bd59-baede55af1cd')

        # All the fields that pyilt2.dataset contains
        setid = mincepy.field()
        setDict = mincepy.field()
        data = mincepy.field()
        headerList = mincepy.field()
        physProps = mincepy.field()
        physUnits = mincepy.field()
        phases = mincepy.field()

    TYPES = (DatasetHelper(),)
