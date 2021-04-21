# -*- coding: utf-8 -*-
# pylint: disable=wrong-import-position
import pytest

pymatgen_core = pytest.importorskip('pymatgen.core')

import mincepy


def test_saving_structure(historian: mincepy.Historian):
    a_lat = 4.03893
    structure = pymatgen_core.Structure(lattice=[[a_lat, 0., 0.], [0., a_lat, 0.], [0., 0., a_lat]],
                                        species=['Al', 'Al', 'Al', 'Al'],
                                        coords=[[0., 0., 0.], [0., 0.5, 0.5], [0.5, 0., 0.5],
                                                [0.5, 0.5, 0.]],
                                        coords_are_cartesian=False)
    structure_id = historian.save(structure)
    del structure

    loaded_structure = historian.load(structure_id)  # type: pymatgen_core.Structure
    assert all(loaded_structure.frac_coords[1] == [0., 0.5, 0.5])
    assert loaded_structure.lattice.a == a_lat
