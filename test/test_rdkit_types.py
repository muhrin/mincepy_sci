# -*- coding: utf-8 -*-
# pylint: disable=wrong-import-order, wrong-import-position

import pytest

pytest.importorskip("rdkit")

import copy

import mincepy
import numpy as np
from rdkit import Chem  # pylint: disable=import-error
import rdkit.Chem.AllChem  # pylint: disable=import-error,unused-import


def create_mol(smiles):
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)
    Chem.AllChem.EmbedMolecule(mol)
    return mol


def test_rdkit_mol(historian: mincepy.Historian):
    smiles = "Cc1ccccc1"
    orig = create_mol(smiles)
    mol = copy.deepcopy(orig)
    oid = historian.save(mol)
    del mol

    loaded = historian.load(oid)
    assert Chem.MolToSmiles(loaded) == Chem.MolToSmiles(orig)
    assert loaded.GetNumAtoms() == orig.GetNumAtoms()
    assert np.allclose(
        orig.GetConformers()[0].GetPositions(),
        loaded.GetConformers()[0].GetPositions(),
        atol=1e-4,
    )
