# -*- coding: utf-8 -*-
"""Module that provides interoperability between pymatgen and mincepy"""
import io
import uuid

import bidict
import mincepy
import torch

# pylint: disable=no-member
torch_dtypes = bidict.bidict(
    {
        "bool": torch.bool,
        "uint8": torch.uint8,
        "int8": torch.int8,
        "int16": torch.int16,
        "int32": torch.int32,
        "int64": torch.int64,
        "float16": torch.float16,
        "float32": torch.float32,
        "float64": torch.float64,
        "complex64": torch.complex64,
        "complex128": torch.complex128,
    }
)


class TensorHelper(mincepy.BaseHelper):
    TYPE = torch.Tensor
    TYPE_ID = uuid.UUID("4b69b98b-3a0e-4d14-8f64-b9bd7caf4cc5")

    DTYPE = "dtype"
    SIZE = "size"
    FILE = "file"

    # pylint: disable=arguments-differ

    def yield_hashables(self, tensor: torch.Tensor, hasher):
        with io.BytesIO() as buffer:
            torch.save(tensor, buffer)
            yield buffer.read()

    def save_instance_state(self, tensor: torch.Tensor, saver: mincepy.Saver):
        tensor_file = saver.historian.create_file("tensor.pt")
        with tensor_file.open("wb") as file:
            torch.save(tensor, file)
        return {
            self.DTYPE: torch_dtypes.inverse[tensor.dtype],
            self.SIZE: tensor.size(),
            self.FILE: tensor_file,
        }

    def new(self, saved_state: dict) -> torch.Tensor:
        return torch.empty(
            saved_state[self.SIZE], dtype=torch_dtypes[saved_state[self.DTYPE]]
        )

    def load_instance_state(self, tensor: torch.Tensor, saved_state: dict, _referencer):
        with saved_state[self.FILE].open("rb") as file:
            tensor[:] = torch.load(file)[:]


class ModuleHelperStub(mincepy.BaseHelper):

    # pylint: disable=arguments-differ

    def yield_hashables(self, module: torch.nn.Module, hasher):
        yield from hasher.yield_hashables(module.__dict__)

    def save_instance_state(self, model: torch.nn.Module, _saver):
        return model.__dict__

    def load_instance_state(
        self, module: torch.nn.Module, saved_state: dict, _referencer
    ):
        for key, value in saved_state.items():
            setattr(module, key, value)


class Conv2dHelper(ModuleHelperStub):
    TYPE = torch.nn.Conv2d
    TYPE_ID = uuid.UUID("bbdb255b-e59b-4968-960d-404f16041379")


class MaxPool2dHelper(ModuleHelperStub):
    TYPE = torch.nn.MaxPool2d
    TYPE_ID = uuid.UUID("eaf92947-98c2-4277-a38a-bc1e30429056")


class LinearHelper(ModuleHelperStub):
    TYPE = torch.nn.Linear
    TYPE_ID = uuid.UUID("2c8c1e39-aa20-4095-81ef-b8e780403e53")


class SavableModuleMixin(mincepy.SavableObject):
    """Mixin for a pytorch module that provides the boilerplate needed to make it savable"""

    # pylint: disable=arguments-differ

    def yield_hashables(self: torch.nn.Module, hasher):
        yield from hasher.yield_hashables(self.__dict__)

    def save_instance_state(self: torch.nn.Module, _saver) -> dict:
        return self.__dict__

    def load_instance_state(self: torch.nn.Module, saved_state: dict, _referencer):
        for key, value in saved_state.items():
            setattr(self, key, value)


class ModuleListHelper(ModuleHelperStub):
    TYPE = torch.nn.ModuleList
    TYPE_ID = uuid.UUID("76fd8263-76c0-4bc4-858e-78e99ad7e332")


class ModuleDictHelper(ModuleHelperStub):
    TYPE = torch.nn.ModuleDict
    TYPE_ID = uuid.UUID("82037c10-7937-4e6f-a6c5-2e022e869186")


TYPES = (
    TensorHelper,
    Conv2dHelper,
    MaxPool2dHelper,
    LinearHelper,
    ModuleListHelper,
    ModuleDictHelper,
)
