"""Module that provides interoperability between pymatgen and mincepy"""

import json
import uuid

import mincepy
from rdkit import Chem
from typing_extensions import override


def get_mol_json_dict(mol: Chem.rdchem.Mol):
    return json.loads(Chem.rdMolInterchange.MolToJSON(mol))


class MolHelper(
    mincepy.BaseHelper,
    obj_type=Chem.rdchem.Mol,
    type_id=uuid.UUID("4810acf6-624c-419f-998c-f1a6dcf9def0"),
):

    @override
    def yield_hashables(self, mol: Chem.rdchem.Mol, hasher, /, *_):
        yield from hasher.yield_hashables(get_mol_json_dict(mol))

    @override
    def save_instance_state(self, mol: Chem.rdchem.Mol, saver, /, *_):
        mol_dict = get_mol_json_dict(mol)
        return mol_dict

    @override
    def new(self, encoded_saved_state: dict, /, *_):
        return Chem.rdMolInterchange.JSONToMols(json.dumps(encoded_saved_state))[0]

    @override
    def load_instance_state(self, mol: Chem.rdchem.Mol, saved_state, _referencer, /):
        pass  # Nothing to do, did it all in new


TYPES = (MolHelper,)
