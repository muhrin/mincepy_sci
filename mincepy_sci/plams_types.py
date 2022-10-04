# -*- coding: utf-8 -*-
import uuid

import mincepy
from scm import plams

# pylint: disable=no-member


class SettingsHelper(mincepy.BaseHelper):
    TYPE = plams.Settings
    TYPE_ID = uuid.UUID("12d88b29-858c-4a12-a5d2-42cb4d4f8ae8")

    def yield_hashables(self, obj, hasher):
        yield from hasher.yield_hashables(obj.as_dict())

    def save_instance_state(
        self, settings: plams.Settings, saver
    ):  # pylint: disable=unused-argument, arguments-differ
        return settings.as_dict()

    def load_instance_state(
        self, settings, saved_state, loader: "mincepy.Loader"
    ):  # pylint: disable=unused-argument, arguments-differ
        """Take the given blank object and load the instance state into it"""
        settings.__init__(**saved_state)


class MoleculeHelper(mincepy.BaseHelper):
    TYPE = plams.Molecule
    TYPE_ID = uuid.UUID("70cafb92-1c0d-4d5f-bf48-5d1c5e70a0ec")

    def yield_hashables(self, obj, hasher):
        yield from hasher.yield_hashables(obj.as_dict())

    def save_instance_state(
        self, molecule: plams.Molecule, saver
    ):  # pylint: disable=unused-argument, arguments-differ
        return molecule.as_dict()

    def load_instance_state(
        self, molecule, saved_state, loader: "mincepy.Loader"
    ):  # pylint: disable=unused-argument, arguments-differ
        """Take the given blank object and load the instance state into it"""

        def constructor():
            return molecule

        plams.Molecule.from_dict.__func__(constructor, saved_state)
        return molecule


TYPES = SettingsHelper(), MoleculeHelper()
