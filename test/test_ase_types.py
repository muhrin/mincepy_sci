# -*- coding: utf-8 -*-
# pylint: disable=wrong-import-position
import pytest

ase = pytest.importorskip("ase")
import ase.build
import mincepy


def test_saving_atoms(historian: mincepy.Historian):
    depth = 2.9
    length = 10.0
    wire = ase.Atoms(
        "Au",
        positions=[[0, length / 2, length / 2]],
        cell=[depth, length, length],
        pbc=[1, 0, 0],
    )

    wire_id = historian.save(wire)
    del wire

    loaded_wire = historian.load(wire_id)  # type: ase.Atoms
    assert all(loaded_wire.pbc == [1, 0, 0])
    assert all(loaded_wire.positions[0] == [0, length / 2, length / 2])


def test_saving_atoms_no_pbc(historian: mincepy.Historian):
    atoms = ase.build.molecule("H2")
    atoms_id = historian.save(atoms)
    del atoms

    assert historian.load(atoms_id).get_chemical_formula() == "H2"
