import uuid

import mincepy
from scm import plams
from typing_extensions import override

# pylint: disable=no-member


class SettingsHelper(
    mincepy.BaseHelper,
    obj_type=plams.Settings,
    type_id=uuid.UUID("12d88b29-858c-4a12-a5d2-42cb4d4f8ae8"),
):
    @override
    def yield_hashables(self, obj, hasher, /):
        yield from hasher.yield_hashables(obj.as_dict())

    @override
    def save_instance_state(self, settings: plams.Settings, saver, /):
        return settings.as_dict()

    @override
    def load_instance_state(self, settings, saved_state, _loader: "mincepy.Loader", /):
        """Take the given blank object and load the instance state into it"""
        settings.__init__(**saved_state)  # pylint: disable=unnecessary-dunder-call


class MoleculeHelper(
    mincepy.BaseHelper,
    obj_type=plams.Molecule,
    type_id=uuid.UUID("70cafb92-1c0d-4d5f-bf48-5d1c5e70a0ec"),
):
    @override
    def yield_hashables(self, obj, hasher, /):
        yield from hasher.yield_hashables(obj.as_dict())

    @override
    def save_instance_state(self, molecule: plams.Molecule, saver, /):
        return molecule.as_dict()

    @override
    def load_instance_state(self, molecule, saved_state, loader: "mincepy.Loader", /):
        """Take the given blank object and load the instance state into it"""

        def constructor():
            return molecule

        plams.Molecule.from_dict.__func__(constructor, saved_state)
        return molecule


class AtomHelper(
    mincepy.TypeHelper,
    obj_type=plams.mol.atom.Atom,
    type_id=uuid.UUID("fbe78755-a8d7-415d-b8b5-44ac18ef5f3a"),
):
    symbol = mincepy.field()
    mol = mincepy.field(ref=True)
    bonds = mincepy.field()
    properties = mincepy.field()
    coords = mincepy.field()


class BondHelper(
    mincepy.TypeHelper,
    obj_type=plams.mol.bond.Bond,
    type_id=uuid.UUID("ad6262eb-8ca1-43a4-bf61-34474cf69137"),
):
    atom1 = mincepy.field(ref=True)
    atom2 = mincepy.field(ref=True)
    order = mincepy.field()
    mol = mincepy.field(ref=True)
    properties = mincepy.field()


class AsArrayContextHelper(
    mincepy.TypeHelper,
    obj_type=plams.mol.context.AsArrayContext,
    type_id=uuid.UUID("4027cba2-f402-484b-9f15-abe6f6979bc3"),
):
    """
    See: https://github.com/SCM-NV/PLAMS/blob/trunk/mol/context.py
    """

    @override
    def yield_hashables(self, obj: plams.mol.context.AsArrayContext, hasher, /):
        # pylint: disable=protected-access
        yield from hasher.yield_hashables(mincepy.ref(obj._from_array.__self__))

    @override
    def eq(
        self,
        one: plams.mol.context.AsArrayContext,
        other: plams.mol.context.AsArrayContext,
        /,  # noqa: W504
    ) -> bool:
        # pylint: disable=protected-access
        return one._from_array.__self__ == other._from_array.__self__

    @override
    def save_instance_state(
        # pylint: disable=unused-argument
        self,
        obj: plams.mol.context.AsArrayContext,
        saver: "mincepy.Saver",
        /,
    ):
        return {"mol": mincepy.ref(obj._from_array.__self__)}  # pylint: disable=protected-access

    @override
    def load_instance_state(
        # pylint: disable=unused-argument
        self,
        obj: plams.mol.context.AsArrayContext,
        saved_state,
        loader: "mincepy.Loader",
        /,
    ):
        # pylint: disable=protected-access
        mol: plams.Molecule = saved_state["mol"]
        obj._atoms = mol.atoms
        obj._from_array = mol.from_array


TYPES = SettingsHelper, MoleculeHelper, AtomHelper, BondHelper, AsArrayContextHelper
