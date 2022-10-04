# -*- coding: utf-8 -*-
"""Module that provides interoperability between pymatgen and mincepy"""

import uuid

# pylint: disable=ungrouped-imports
import collections.abc
import mincepy
import numpy
import pymatgen.core
from pymatgen.electronic_structure.core import Spin
import pymatgen.electronic_structure.bandstructure as pymatgen_bandstructure
import pymatgen.electronic_structure.dos as pymatgen_dos


def _clean_recursive(obj):
    if isinstance(obj, collections.abc.Mapping):
        clean_obj = {key: _clean_recursive(value) for key, value in obj.items()}
    elif isinstance(obj, collections.abc.Sequence) and not isinstance(obj, str):
        clean_obj = [_clean_recursive(item) for item in obj]
    elif type(obj).__module__ == "numpy":
        clean_obj = obj.tolist()
    else:
        clean_obj = obj
    return clean_obj


class StructureHelper(mincepy.TypeHelper):
    TYPE = pymatgen.core.Structure
    TYPE_ID = uuid.UUID("b00aa5f5-f152-43c9-aeab-9710b0f045b1")
    INJECT_CREATION_TRACKING = True

    def yield_hashables(
        self, structure: pymatgen.core.Structure, hasher
    ):  # pylint: disable=arguments-differ
        yield from hasher.yield_hashables(structure.as_dict())

    def eq(self, one, other) -> bool:
        return (
            one == other
        )  # pymatgen.core.IStructure defines __eq__, which Structure inherits

    def save_instance_state(
        self, structure: pymatgen.core.Structure, _referencer
    ):  # pylint: disable=arguments-differ
        return structure.as_dict()

    def new(self, encoded_saved_state):
        return pymatgen.core.Structure.from_dict(encoded_saved_state)

    def load_instance_state(
        self, structure: pymatgen.core.Structure, saved_state, _referencer
    ):  # pylint: disable=arguments-differ
        pass  # Nothing to do, did it all in new


class MoleculeHelper(mincepy.TypeHelper):
    TYPE = pymatgen.core.Molecule
    TYPE_ID = uuid.UUID("d96f94c2-bc63-4abc-9e75-9d297b0ca9ad")

    def yield_hashables(
        self, mol: pymatgen.core.Molecule, hasher
    ):  # pylint: disable=arguments-differ
        yield from hasher.yield_hashables(mol.as_dict())

    def eq(self, one, other) -> bool:
        return one == other

    def save_instance_state(
        self, mol: pymatgen.core.Molecule, _referencer
    ):  # pylint: disable=arguments-differ
        return mol.as_dict()

    def new(self, encoded_saved_state):
        return pymatgen.core.Molecule.from_dict(encoded_saved_state)

    def load_instance_state(
        self, structure: pymatgen.core.Molecule, saved_state, _referencer
    ):  # pylint: disable=arguments-differ
        pass  # Nothing to do, did it all in new


class BandStructureHelper(mincepy.TypeHelper):
    TYPE = pymatgen_bandstructure.BandStructure
    TYPE_ID = uuid.UUID("690b9a99-3f1f-45e5-88eb-0448ceaff7dd")
    INJECT_CREATION_TRACKING = True

    def yield_hashables(
        self, bandstructure: pymatgen_bandstructure.BandStructure, hasher
    ):  # pylint: disable=arguments-differ
        yield from hasher.yield_hashables(bandstructure.as_dict())

    def eq(self, one, other) -> bool:
        # pylint: disable=too-many-return-statements
        if not (
            isinstance(one, pymatgen_bandstructure.BandStructure)
            and isinstance(other, pymatgen_bandstructure.BandStructure)
        ):
            return False

        # Check the cheap things first
        # pylint: disable=too-many-boolean-expressions
        if (
            (len(one.kpoints) != len(other.kpoints))
            or (one.lattice != other.lattice)
            or (one.efermi != other.efermi)
            or (one.is_spin_polarized != other.is_spin_polarized)
            or (one.nb_bands != other.nb_bands)
            or (one.structure != other.structure)
            or (len(one.projections) != len(other.projections))
        ):
            return False

        # Check expensive things one-by-one
        if not (
            numpy.array([kpoint.cart_coords for kpoint in one.kpoints])
            == numpy.array([kpoint.cart_coords for kpoint in other.kpoints])
        ).all():
            return False

        if not (
            numpy.array(one.bands.get(Spin.up, []))
            == numpy.array(other.bands.get(Spin.up, []))
        ).all():
            return False

        if not (
            numpy.array(one.bands.get(Spin.down, []))
            == numpy.array(other.bands.get(Spin.down, []))
        ).all():
            return False

        if not (
            numpy.array(one.projections.get(Spin.up, []))
            == numpy.array(other.projections.get(Spin.up, []))
        ).all():
            return False

        if not (
            numpy.array(one.projections.get(Spin.down, []))
            == numpy.array(other.projections.get(Spin.down, []))
        ).all():
            return False

        return True

    # pylint: disable=arguments-differ
    def save_instance_state(
        self, bandstructure: pymatgen_bandstructure.BandStructure, _referencer
    ):

        # The `pymatgen.electronic_structure.bandstructure.Kpoints.as_dict` method uses
        # `list(numpy.array)` instead of `numpy.array.tolist()`, so there are numpy
        # numbers in the 'kpoints' and 'labels_dict' members of `BandStructure.as_dict`
        # dictionaries, which need to be converted to python types. This issue is fixed
        # in materialsproject/pymatgen #2113.
        return _clean_recursive(bandstructure.as_dict())

    def new(self, encoded_saved_state):
        return pymatgen_bandstructure.BandStructure.from_dict(encoded_saved_state)

    # pylint: disable=arguments-differ
    def load_instance_state(
        self,
        bandstructure: pymatgen_bandstructure.BandStructure,
        saved_state,
        _referencer,
    ):
        pass  # Nothing to do, did it all in new


class CompleteDosHelper(mincepy.TypeHelper):
    TYPE = pymatgen_dos.CompleteDos
    TYPE_ID = uuid.UUID("cf98144c-59e0-4235-8faa-3dd883651c6a")
    INJECT_CREATION_TRACKING = True

    def yield_hashables(
        self, completedos: pymatgen_dos.CompleteDos, hasher
    ):  # pylint: disable=arguments-differ
        yield from hasher.yield_hashables(completedos.as_dict())

    def eq(self, one, other) -> bool:
        # pylint: disable=too-many-boolean-expressions
        if not (
            isinstance(one, pymatgen_dos.CompleteDos)
            and isinstance(other, pymatgen_dos.CompleteDos)
        ):
            return False

        if (
            (one.structure == other.structure)
            and (one.efermi == other.efermi)
            and (
                numpy.array(one.get_cbm_vbm()) == numpy.array(other.get_cbm_vbm())
            ).all()
            and (one.energies == other.energies).all()
            and (
                numpy.array(one.densities.get(Spin.up, []))
                == numpy.array(other.densities.get(Spin.up, []))
            ).all()
            and (
                numpy.array(one.densities.get(Spin.down, []))
                == numpy.array(other.densities.get(Spin.down, []))
            ).all()
        ):
            return True

        return False

    def save_instance_state(
        self, completedos: pymatgen_dos.CompleteDos, _referencer
    ):  # pylint: disable=arguments-differ
        return _clean_recursive(completedos.as_dict())

    def new(self, encoded_saved_state):
        return pymatgen_dos.CompleteDos.from_dict(encoded_saved_state)

    # pylint: disable=arguments-differ
    def load_instance_state(
        self, completedos: pymatgen_dos.CompleteDos, saved_state, _referencer
    ):
        pass  # Nothing to do, did it all in new


class PeriodicSite(mincepy.TypeHelper):
    TYPE = pymatgen.core.PeriodicSite
    TYPE_ID = uuid.UUID("24ddfbb3-c3e6-432f-abd8-4542810ac002")
    INJECT_CREATION_TRACKING = True

    def yield_hashables(
        self, site: pymatgen.core.PeriodicSite, hasher
    ):  # pylint: disable=arguments-differ
        yield from hasher.yield_hashables(site.as_dict())

    def eq(self, one, other) -> bool:
        # pylint: disable=too-many-boolean-expressions
        if not (
            isinstance(one, pymatgen.core.PeriodicSite)
            and isinstance(other, pymatgen.core.PeriodicSite)
        ):
            return False

        return one == other  # Piggypack off the __eq__

    def save_instance_state(
        self, site: pymatgen.core.PeriodicSite, _referencer
    ):  # pylint: disable=arguments-differ
        return _clean_recursive(site.as_dict())

    def new(self, encoded_saved_state):
        return pymatgen.core.PeriodicSite.from_dict(encoded_saved_state)

    # pylint: disable=arguments-differ
    def load_instance_state(
        self, site: pymatgen.core.PeriodicSite, saved_state, _referencer
    ):
        pass  # Nothing to do, did it all in new


TYPES = (
    StructureHelper(),
    MoleculeHelper(),
    BandStructureHelper(),
    CompleteDosHelper(),
    PeriodicSite(),
)
