"""Module that provides interoperability between pymatgen and mincepy"""

import copy
import pickle  # nosec
import uuid

from e3nn import math, nn, o3
import mincepy
import torch
from typing_extensions import override

from . import pytorch_types


class IrrepHelper(
    mincepy.BaseHelper,
    obj_type=o3.Irrep,
    type_id=uuid.UUID("7a525788-8f67-4575-a1e4-9a68195516e2"),
    immutable=True,
):
    @override
    def yield_hashables(self, irrep: o3.Irrep, hasher, /):
        yield from hasher.yield_hashables(str(irrep))

    @override
    def save_instance_state(self, irrep: o3.Irrep, saver: mincepy.Saver, /):
        return str(irrep)

    @override
    def new(self, saved_state: str, /) -> o3.Irrep:
        return o3.Irrep(saved_state)

    @override
    def load_instance_state(self, irrep: o3.Irrep, saved_state: str, _referencer, /):
        # Nothing to do, all done in new()
        pass


class IrrepsHelper(
    mincepy.BaseHelper,
    obj_type=o3.Irreps,
    type_id=uuid.UUID("bfc8a923-a316-4b5d-9b05-98efe6e7d7fb"),
    immutable=True,
):
    @override
    def yield_hashables(self, irreps: o3.Irreps, hasher, /):
        yield from hasher.yield_hashables(str(irreps))

    @override
    def save_instance_state(self, irreps: o3.Irreps, saver: mincepy.Saver, /):
        return str(irreps)

    @override
    def new(self, saved_state: str, /) -> o3.Irreps:
        return o3.Irreps(saved_state)

    @override
    def load_instance_state(self, irreps: o3.Irreps, saved_state: str, _referencer, /):
        # Nothing to do, all done in new()
        pass


class CodeGenMixinHelper(mincepy.TypeHelper, obj_type=None, type_id=None):
    """A stub type helper that can be used for modules that use the CodeGenMixin"""

    @override
    def yield_hashables(self, rtp: o3.ReducedTensorProducts, hasher, /):
        yield from hasher.yield_hashables(rtp.__getstate__())

    @override
    def save_instance_state(self, rtp: o3.ReducedTensorProducts, saver: mincepy.Saver, /) -> dict:
        return rtp.__getstate__()

    @override
    def load_instance_state(self, rtp: o3.ReducedTensorProducts, saved_state: dict, _referencer, /):
        rtp.__setstate__(saved_state)


# region o3


class InstructionHelper(
    mincepy.common_helpers.TupleHelper,
    obj_type=o3.Instruction,
    type_id=uuid.UUID("a76e6544-39aa-4291-89ab-aa7c48c97484"),
):
    @override
    def save_instance_state(self, obj: tuple, /, *_):
        return list(obj)

    @override
    def new(self, encoded_saved_state, /):
        return self.TYPE(*encoded_saved_state)


class TensorProductHelper(
    CodeGenMixinHelper,
    obj_type=o3.TensorProduct,
    type_id=uuid.UUID("1a9a6154-ba68-476b-bb09-2ace5fda5f45"),
):

    # Helpers to make finding easier
    irreps_in1 = mincepy.field("irreps_in1")
    irreps_in2 = mincepy.field("irreps_in2")
    irreps_out = mincepy.field("irreps_out")


class TensorSquareHelper(
    CodeGenMixinHelper,
    obj_type=o3.TensorSquare,
    type_id=uuid.UUID("aa68e071-92ee-48ad-8909-80f9069f629c"),
):

    # Helpers to make finding easier
    irreps_in = mincepy.field("irreps_in")


class ReducedTensorProductsHelper(
    CodeGenMixinHelper,
    obj_type=o3.ReducedTensorProducts,
    type_id=uuid.UUID("b87bf7b1-eb3d-4d5d-b65a-3a6780d725a3"),
):

    # Helpers to make finding easier
    irreps_in = mincepy.field("irreps_in")
    irreps_out = mincepy.field("irreps_out")


class ElementwiseTensorProductHelper(
    TensorProductHelper,
    obj_type=o3.ElementwiseTensorProduct,
    type_id=uuid.UUID("5c808ac0-0e37-4079-976d-862d111f897b"),
):
    pass


# endregion


# region nn
class GateHelper(
    pytorch_types.ModuleHelperStub,
    obj_type=nn.Gate,
    type_id=uuid.UUID("ad05bad7-aa30-4e81-8f61-308640db1878"),
):
    irreps_in = mincepy.field("irreps_in")
    irreps_out = mincepy.field("irreps_out")


class ShortcutHelper(
    pytorch_types.ModuleHelperStub,
    obj_type=nn._gate._Sortcut,  # pylint: disable=protected-access
    type_id=uuid.UUID("c1de96e4-0963-4528-8d8e-87711049cc7b"),
):
    pass


class ExtractHelper(
    CodeGenMixinHelper,
    obj_type=nn.Extract,
    type_id=uuid.UUID("5b2e108e-744a-4537-909b-31826ca6788a"),
):
    pass


class ActivationHelper(
    pytorch_types.ModuleHelperStub,
    obj_type=nn.Activation,
    type_id=uuid.UUID("ee7b6794-d76f-4ac3-bd44-406bc8a9aa1b"),
):
    pass


# endregion


# region math
class Normalize2MomHelper(
    pytorch_types.ModuleHelperStub,
    obj_type=math.normalize2mom,
    type_id=uuid.UUID("fd535825-95ee-42e2-9ce6-0e7a6f43a0ba"),
):
    """We need a little custom saving code here because normalize2mom contains a reference to a
    function (self.f) which mincepy cannot serialise correctly."""

    @override
    def yield_hashables(self, module: torch.nn.Module, hasher, /):
        yield from hasher.yield_hashables(self.cleanup_state(module.__dict__))

    @override
    def save_instance_state(self, model: torch.nn.Module, _saver, /):
        return self.cleanup_state(super().save_instance_state(model, _saver))

    @override
    def load_instance_state(self, module: torch.nn.Module, saved_state: dict, _referencer, /):
        if not isinstance(saved_state["f"], torch.nn.Module):
            loaded = pickle.loads(saved_state["f"])  # nosec
            saved_state["f"] = loaded
        return super().load_instance_state(module, saved_state, _referencer)

    def cleanup_state(self, state_dict: dict):
        if not isinstance(state_dict["f"], torch.nn.Module):
            state_dict = copy.deepcopy(state_dict)
            state_dict["f"] = pickle.dumps(state_dict["f"])
        return state_dict


# endregion

TYPES = (
    # o3
    IrrepHelper,
    IrrepsHelper,
    InstructionHelper,
    TensorProductHelper,
    TensorSquareHelper,
    ReducedTensorProductsHelper,
    ElementwiseTensorProductHelper,
    # nn
    GateHelper,
    ShortcutHelper,
    ExtractHelper,
    ActivationHelper,
    # math
    Normalize2MomHelper,
)
