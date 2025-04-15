"""Module that provides interoperability between pymatgen and mincepy"""

import io
import uuid

import bidict
import mincepy
import torch
from typing_extensions import override

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


class TensorHelper(
    mincepy.BaseHelper,
    obj_type=torch.Tensor,
    type_id=uuid.UUID("4b69b98b-3a0e-4d14-8f64-b9bd7caf4cc5"),
):

    DTYPE = "dtype"
    SIZE = "size"
    FILE = "file"

    @override
    def yield_hashables(self, tensor: torch.Tensor, hasher, /):
        with io.BytesIO() as buffer:
            torch.save(tensor, buffer)
            yield buffer.read()

    @override
    def save_instance_state(self, tensor: torch.Tensor, saver: mincepy.Saver, /):
        tensor_file = saver.historian.create_file("tensor.pt")
        with tensor_file.open("wb") as file:
            torch.save(tensor, file)
        return {
            self.DTYPE: torch_dtypes.inverse[tensor.dtype],
            self.SIZE: tensor.size(),
            self.FILE: tensor_file,
        }

    @override
    def new(self, saved_state: dict, /) -> torch.Tensor:
        return torch.empty(saved_state[self.SIZE], dtype=torch_dtypes[saved_state[self.DTYPE]])

    @override
    def load_instance_state(self, tensor: torch.Tensor, saved_state: dict, _referencer, /):
        with saved_state[self.FILE].open("rb") as file:
            tensor[:] = torch.load(file)[:]  # nosec


class ModuleHelperStub(mincepy.BaseHelper, obj_type=None, type_id=None):
    @override
    def yield_hashables(self, module: torch.nn.Module, hasher, /):
        yield from hasher.yield_hashables(module.__dict__)

    @override
    def save_instance_state(self, model: torch.nn.Module, _saver, /):
        return model.__dict__

    @override
    def load_instance_state(self, module: torch.nn.Module, saved_state: dict, _referencer, /):
        for key, value in saved_state.items():
            setattr(module, key, value)


class Conv2dHelper(
    ModuleHelperStub,
    obj_type=torch.nn.Conv2d,
    type_id=uuid.UUID("bbdb255b-e59b-4968-960d-404f16041379"),
):
    pass


class MaxPool2dHelper(
    ModuleHelperStub,
    obj_type=torch.nn.MaxPool2d,
    type_id=uuid.UUID("eaf92947-98c2-4277-a38a-bc1e30429056"),
):
    pass


class LinearHelper(
    ModuleHelperStub,
    obj_type=torch.nn.Linear,
    type_id=uuid.UUID("2c8c1e39-aa20-4095-81ef-b8e780403e53"),
):
    pass


class SavableModuleMixin(mincepy.SavableObject):
    """Mixin for a pytorch module that provides the boilerplate needed to make it savable"""

    @override
    def yield_hashables(self: torch.nn.Module, hasher, /):
        yield from hasher.yield_hashables(self.__dict__)

    @override
    def save_instance_state(self: torch.nn.Module, _saver, /) -> dict:
        return self.__dict__

    @override
    def load_instance_state(self: torch.nn.Module, saved_state: dict, _referencer, /):
        for key, value in saved_state.items():
            setattr(self, key, value)


class ModuleListHelper(
    ModuleHelperStub,
    obj_type=torch.nn.ModuleList,
    type_id=uuid.UUID("76fd8263-76c0-4bc4-858e-78e99ad7e332"),
):
    pass


class ModuleDictHelper(
    ModuleHelperStub,
    obj_type=torch.nn.ModuleDict,
    type_id=uuid.UUID("82037c10-7937-4e6f-a6c5-2e022e869186"),
):
    pass


class ParameterHelper(
    mincepy.TypeHelper,
    obj_type=torch.nn.Parameter,
    type_id=uuid.UUID("834d370b-288c-4d77-9249-e645ad6bc0a5"),
):
    data = mincepy.field(type=torch.Tensor)
    requires_grad = mincepy.field(type=bool)


class SizeHelper(
    mincepy.builtins.TupleHelper,
    obj_type=torch.Size,
    type_id=uuid.UUID("f0d9cdfc-b2e0-4d68-b288-b11c19ab464f"),
    immutable=True,
):
    pass


TYPES = (
    TensorHelper,
    Conv2dHelper,
    MaxPool2dHelper,
    LinearHelper,
    ModuleListHelper,
    ModuleDictHelper,
    ParameterHelper,
    SizeHelper,
)
