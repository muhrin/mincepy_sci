# -*- coding: utf-8 -*-
"""Module that provides interoperability between pymatgen and mincepy"""
try:
    import pymatgen.core
except ImportError:
    TYPES = tuple()
else:
    import uuid

    import mincepy

    class StructureHelper(mincepy.TypeHelper):
        TYPE = pymatgen.core.Structure
        TYPE_ID = uuid.UUID('b00aa5f5-f152-43c9-aeab-9710b0f045b1')
        INJECT_CREATION_TRACKING = True

        def yield_hashables(self, structure: pymatgen.core.Structure, hasher):  # pylint: disable=arguments-differ
            yield from hasher.yield_hashables(structure.as_dict())

        def eq(self, one, other) -> bool:
            return one == other  # pymatgen.core.IStructure defines __eq__, which Structure inherits

        def save_instance_state(self, structure: pymatgen.core.Structure, _referencer):  # pylint: disable=arguments-differ
            return structure.as_dict()

        def new(self, encoded_saved_state):
            return pymatgen.core.Structure.from_dict(encoded_saved_state)

        def load_instance_state(self, structure: pymatgen.core.Structure, saved_state, _referencer):  # pylint: disable=arguments-differ
            pass  # Nothing to do, did it all in new

    TYPES = (StructureHelper(),)
