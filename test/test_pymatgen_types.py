# -*- coding: utf-8 -*-
# pylint: disable=wrong-import-position
import json
import pathlib

import pytest

pymatgen_core = pytest.importorskip("pymatgen.core")
pymatgen_electronic_structure = pytest.importorskip(
    "pymatgen.electronic_structure.core"
)
pymatgen_bandstructure = pytest.importorskip(
    "pymatgen.electronic_structure.bandstructure"
)
pymatgen_dos = pytest.importorskip("pymatgen.electronic_structure.dos")

import mincepy
import numpy


def test_saving_structure(historian: mincepy.Historian):
    a_lat = 4.03893
    structure = pymatgen_core.Structure(
        lattice=[[a_lat, 0.0, 0.0], [0.0, a_lat, 0.0], [0.0, 0.0, a_lat]],
        species=["Al", "Al", "Al", "Al"],
        coords=[[0.0, 0.0, 0.0], [0.0, 0.5, 0.5], [0.5, 0.0, 0.5], [0.5, 0.5, 0.0]],
        coords_are_cartesian=False,
    )
    structure_id = historian.save(structure)
    del structure

    loaded_structure: pymatgen_core.Structure = historian.load(structure_id)
    assert all(loaded_structure.frac_coords[1] == [0.0, 0.5, 0.5])
    assert numpy.all(loaded_structure.lattice.a == a_lat)


def test_saving_molecule(historian: mincepy.Historian):
    coords = [
        [0.000000, 0.000000, 0.000000],
        [0.000000, 0.000000, 1.089000],
        [1.026719, 0.000000, -0.363000],
        [-0.513360, -0.889165, -0.363000],
        [-0.513360, 0.889165, -0.363000],
    ]
    methane = pymatgen_core.Molecule(["C", "H", "H", "H", "H"], coords)
    mol_id = historian.save(methane)
    del methane

    loaded_mol: pymatgen_core.Molecule = historian.load(mol_id)
    assert numpy.all(loaded_mol.cart_coords == coords)


def test_saving_bandstructure(historian: mincepy.Historian):
    a_lat = 4.03893
    structure = pymatgen_core.Structure(
        lattice=[[a_lat, 0.0, 0.0], [0.0, a_lat, 0.0], [0.0, 0.0, a_lat]],
        species=["Al", "Al", "Al", "Al"],
        coords=[[0.0, 0.0, 0.0], [0.0, 0.5, 0.5], [0.5, 0.0, 0.5], [0.5, 0.5, 0.0]],
        coords_are_cartesian=False,
    )
    lattice = structure.lattice
    kpoints = [numpy.array([0.0, 0.0, 0.0])]
    eigenvals = {
        pymatgen_electronic_structure.Spin.up: numpy.array(
            [[0.0], [1.0], [2.0], [3.0]]
        ),
        pymatgen_electronic_structure.Spin.down: numpy.array(
            [[0.5], [1.5], [2.5], [3.5]]
        ),
    }
    efermi = 1.75
    labels_dict = {"gamma": [0.0, 0.0, 0.0]}
    bandstructure = pymatgen_bandstructure.BandStructure(
        kpoints=kpoints,
        eigenvals=eigenvals,
        lattice=lattice,
        efermi=efermi,
        labels_dict=labels_dict,
        coords_are_cartesian=False,
        structure=structure,
        projections=None,
    )

    bandstructure_id = historian.save(bandstructure)
    del bandstructure

    loaded_bandstructure = historian.load(
        bandstructure_id
    )  # type: pymatgen_bandstructure.BandStructure
    assert (
        loaded_bandstructure.kpoints[0].frac_coords == numpy.array([0.0, 0.0, 0.0])
    ).all()


@pytest.fixture()
def completedos_json_dict():
    dospath = pathlib.Path(__file__).parent / "res/mp-148_Si_CompleteDos.json"
    with open(str(dospath), "r") as stream:
        return json.load(stream)


def test_saving_completedos(
    historian: mincepy.Historian, completedos_json_dict
):  # pylint: disable=redefined-outer-name
    completedos = pymatgen_dos.CompleteDos.from_dict(completedos_json_dict)
    completedos_id = historian.save(completedos)
    del completedos

    loaded_completedos = historian.load(
        completedos_id
    )  # type: pymatgen_dos.CompleteDos
    assert loaded_completedos.efermi == 3.93446269
    assert loaded_completedos.energies[0] == -24.7685


def test_periodic_site(historian: mincepy.Historian):
    lattice = pymatgen_core.Lattice.from_parameters(1.0, 1.0, 1.0, 85.0, 52.0, 124.0)
    site = pymatgen_core.PeriodicSite("Li", coords=[0.2, 0.56, 0.235], lattice=lattice)

    site_id = historian.save(site)
    del site
    loaded_site = historian.load(site_id)  # type: pymatgen_core.PeriodicSite
    assert loaded_site.a == 0.2
    assert loaded_site.b == 0.56
    assert loaded_site.c == 0.235
    assert loaded_site.lattice == lattice
