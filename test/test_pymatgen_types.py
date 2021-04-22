# -*- coding: utf-8 -*-
# pylint: disable=wrong-import-position
import pytest

pymatgen_core = pytest.importorskip('pymatgen.core')
pymatgen_electronic_structure = pytest.importorskip('pymatgen.electronic_structure.core')
pymatgen_bandstructure = pytest.importorskip('pymatgen.electronic_structure.bandstructure')
pymatgen_dos = pytest.importorskip('pymatgen.electronic_structure.dos')

import mincepy
import numpy


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


def test_saving_bandstructure(historian: mincepy.Historian):
    a_lat = 4.03893
    structure = pymatgen_core.Structure(lattice=[[a_lat, 0., 0.], [0., a_lat, 0.], [0., 0., a_lat]],
                                        species=['Al', 'Al', 'Al', 'Al'],
                                        coords=[[0., 0., 0.], [0., 0.5, 0.5], [0.5, 0., 0.5],
                                                [0.5, 0.5, 0.]],
                                        coords_are_cartesian=False)
    lattice = structure.lattice
    kpoints = [numpy.array([0., 0., 0.])]
    eigenvals = {
        pymatgen_electronic_structure.Spin.up: numpy.array([[0.], [1.], [2.], [3.]]),
        pymatgen_electronic_structure.Spin.down: numpy.array([[0.5], [1.5], [2.5], [3.5]])
    }
    efermi = 1.75
    labels_dict = {'gamma': [0., 0., 0.]}
    bandstructure = pymatgen_bandstructure.BandStructure(kpoints=kpoints,
                                                         eigenvals=eigenvals,
                                                         lattice=lattice,
                                                         efermi=efermi,
                                                         labels_dict=labels_dict,
                                                         coords_are_cartesian=False,
                                                         structure=structure,
                                                         projections=None)

    bandstructure_id = historian.save(bandstructure)
    del bandstructure

    loaded_bandstructure = historian.load(
        bandstructure_id)  # type: pymatgen_bandstructure.BandStructure
    assert (loaded_bandstructure.kpoints[0].frac_coords == numpy.array([0., 0., 0.])).all()
