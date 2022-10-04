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
    """This stub needs to be implemented for each model type that you want to load with the
    TYPE and TYPE_ID set uniquely"""

    # pylint: disable=arguments-differ

    def yield_hashables(self, module: torch.nn.Module, hasher):
        yield from hasher.yield_hashables(module.state_dict())

    def save_instance_state(self, model: torch.nn.Module, _saver):
        return dict(model.state_dict())

    def load_instance_state(
        self, module: torch.nn.Module, saved_state: dict, _referencer
    ):
        # Need to call at least the Module constructor so that it creates some necessary members
        module.__init__()
        module.load_state_dict(saved_state)
        module.eval()


class SavableModuleMixin(mincepy.SavableObject):
    """Mixin for a pytorch module that provides the boilerplate needed to make it savable"""

    # pylint: disable=arguments-differ

    def yield_hashables(self: torch.nn.Module, hasher):
        yield from hasher.yield_hashables(self.state_dict())

    def save_instance_state(self: torch.nn.Module, _saver) -> dict:
        return dict(self.state_dict())

    def load_instance_state(self: torch.nn.Module, saved_state: dict, _referencer):
        self.__init__()
        self.load_state_dict(saved_state)
        self.eval()


TYPES = (TensorHelper(),)
