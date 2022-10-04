# -*- coding: utf-8 -*-
# pylint: disable=wrong-import-position
import copy
import uuid

import pytest

torch = pytest.importorskip("torch")

import mincepy
from torch import nn  # pylint: disable=import-error

from mincepy_sci import pytorch_types


def test_saving_tensor(historian: mincepy.Historian):
    tensor = torch.rand([10, 5])
    data = tensor.tolist()
    tensor_id = historian.save(tensor)
    del tensor

    loaded = historian.load(tensor_id)
    assert torch.all(torch.tensor(data) == loaded)


def test_saving_module_inheritance(historian: mincepy.Historian):
    """Test saving and loading module using inheritance.

    Example from:
    https://pytorch.org/tutorials/beginner/saving_loading_models.html"""
    model = TheSavableModelClass()
    state_dict = copy.deepcopy(model.state_dict())  # pylint: disable=missing-kwoa

    model_id = historian.save(model)
    del model

    loaded = historian.load(model_id)
    _compare_dicts(state_dict, loaded.state_dict())


def test_saving_module_type_helper(historian: mincepy.Historian):
    """Test saving and loading module using type helper.

    Example from:
    https://pytorch.org/tutorials/beginner/saving_loading_models.html"""
    historian.register_type(TheModelClassHelper)

    model = TheModelClass()
    state_dict = copy.deepcopy(model.state_dict())  # pylint: disable=missing-kwoa

    model_id = historian.save(model)
    del model

    loaded = historian.load(model_id)
    _compare_dicts(state_dict, loaded.state_dict())


def _compare_dicts(dict1, dict2):
    assert set(dict1.keys()) == set(dict2.keys())
    for key in dict1.keys():
        assert torch.all(dict1[key] == dict2[key])


class TheModelClass(nn.Module):
    def __init__(self):
        super(TheModelClass, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, val):
        val = self.pool(torch.F.relu(self.conv1(val)))
        val = self.pool(torch.F.relu(self.conv2(val)))
        val = val.view(-1, 16 * 5 * 5)
        val = torch.F.relu(self.fc1(val))
        val = torch.F.relu(self.fc2(val))
        val = self.fc3(val)
        return val


class TheModelClassHelper(pytorch_types.ModuleHelperStub):
    TYPE = TheModelClass
    TYPE_ID = uuid.UUID("1df0f5aa-42a2-4fb5-8efc-f03339dfe306")


class TheSavableModelClass(TheModelClass, pytorch_types.SavableModuleMixin):
    TYPE_ID = uuid.UUID("1df0f5aa-42a2-4fb5-8efc-f03339dfe306")
