# -*- coding: utf-8 -*-
# pylint: disable=wrong-import-position, invalid-name
import pytest

numpy = pytest.importorskip("scm.plams")

import mincepy

# pylint: disable=no-member


def test_saving_plams_settings(historian: mincepy.Historian):
    from scm import plams

    settings = plams.Settings()
    settings.sub.a = "a"
    settings.sub.b = "b"
    settings.top = "top"

    settings_id = historian.save(settings)
    del settings

    loaded = historian.load(settings_id)
    assert isinstance(loaded.sub, plams.Settings)
    assert loaded.sub.a == "a"
    assert loaded.sub.b == "b"
    assert loaded.top == "top"


def test_saving_plams_molecule(historian: mincepy.Historian):
    from scm import plams

    atoms_list = (
        plams.Atom(symbol="O", coords=(0, 0, 0)),
        plams.Atom(symbol="H", coords=(1, 0, 0)),
        plams.Atom(symbol="H", coords=(0, 1, 0)),
    )
    atoms_list[0].properties.my_favourite = True

    mol = plams.Molecule()
    mol.properties.bangin = True
    for entry in atoms_list:
        mol.add_atom(entry)
    mol.add_bond(mol[1], mol[2], order=plams.Bond.AR)

    mol_id = historian.save(mol)
    del mol

    loaded = historian.load(mol_id)
    # Check that atoms are as we expect
    for atom, entry in zip(loaded, atoms_list):
        assert atom_eq(atom, entry)
    assert loaded.properties.bangin


def atom_eq(atom1, atom2):
    for attr in ("atnum", "coords", "properties"):
        if getattr(atom1, attr) != getattr(atom2, attr):
            return False

    return True
