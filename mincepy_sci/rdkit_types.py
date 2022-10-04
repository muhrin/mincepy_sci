# -*- coding: utf-8 -*-
"""Module that provides interoperability between pymatgen and mincepy"""
import json
import uuid

import mincepy
from rdkit import Chem


def get_mol_json_dict(mol: Chem.rdchem.Mol):
    return json.loads(Chem.rdMolInterchange.MolToJSON(mol))


class MolHelper(mincepy.BaseHelper):
    TYPE = Chem.rdchem.Mol
    TYPE_ID = uuid.UUID("4810acf6-624c-419f-998c-f1a6dcf9def0")

    # pylint: disable=arguments-differ

    def yield_hashables(self, mol: Chem.rdchem.Mol, hasher):
        yield from hasher.yield_hashables(get_mol_json_dict(mol))

    def save_instance_state(self, mol: Chem.rdchem.Mol, saver):
        mol_dict = get_mol_json_dict(mol)
        return mol_dict

    def new(self, encoded_saved_state: dict):
        return Chem.rdMolInterchange.JSONToMols(json.dumps(encoded_saved_state))[0]

    def load_instance_state(
        self, mol: Chem.rdchem.Mol, saved_state, _referencer
    ):  # pylint: disable=arguments-differ
        pass  # Nothing to do, did it all in new


TYPES = (MolHelper(),)
