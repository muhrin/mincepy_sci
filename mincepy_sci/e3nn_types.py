# -*- coding: utf-8 -*-
"""Module that provides interoperability between pymatgen and mincepy"""
import copy
import pickle  # nosec
import uuid

from e3nn import math
from e3nn import nn
from e3nn import o3
import mincepy
import torch

from . import pytorch_types


class IrrepHelper(mincepy.BaseHelper):
    TYPE = o3.Irrep
    TYPE_ID = uuid.UUID("7a525788-8f67-4575-a1e4-9a68195516e2")
    IMMUTABLE = True

    # pylint: disable=arguments-differ

    def yield_hashables(self, irrep: o3.Irrep, hasher):
        yield from hasher.yield_hashables(str(irrep))

    def save_instance_state(self, irrep: o3.Irrep, saver: mincepy.Saver):
        return str(irrep)

    def new(self, saved_state: str) -> o3.Irrep:
        return o3.Irrep(saved_state)

    def load_instance_state(self, irrep: o3.Irrep, saved_state: str, _referencer):
        # Nothing to do, all done in new()
        pass


class IrrepsHelper(mincepy.BaseHelper):
    TYPE = o3.Irreps
    TYPE_ID = uuid.UUID("bfc8a923-a316-4b5d-9b05-98efe6e7d7fb")
    IMMUTABLE = True

    # pylint: disable=arguments-differ

    def yield_hashables(self, irreps: o3.Irreps, hasher):
        yield from hasher.yield_hashables(str(irreps))

    def save_instance_state(self, irreps: o3.Irreps, saver: mincepy.Saver):
        return str(irreps)

    def new(self, saved_state: str) -> o3.Irreps:
        return o3.Irreps(saved_state)

    def load_instance_state(self, irreps: o3.Irreps, saved_state: str, _referencer):
        # Nothing to do, all done in new()
        pass


class CodeGenMixinHelper(mincepy.TypeHelper):
    """A stub type helper that can be used for modules that use the CodeGenMixin"""

    # pylint: disable=arguments-differ

    def yield_hashables(self, rtp: o3.ReducedTensorProducts, hasher):
        yield from hasher.yield_hashables(rtp.__getstate__())

    def save_instance_state(
        self, rtp: o3.ReducedTensorProducts, saver: mincepy.Saver
    ) -> dict:
        return rtp.__getstate__()

    def load_instance_state(
        self, rtp: o3.ReducedTensorProducts, saved_state: dict, _referencer
    ):
        rtp.__setstate__(saved_state)


# region o3


class InstructionHelper(mincepy.common_helpers.TupleHelper):
    TYPE = o3.Instruction
    TYPE_ID = uuid.UUID("a76e6544-39aa-4291-89ab-aa7c48c97484")

    # pylint: disable=arguments-differ

    def save_instance_state(self, obj: tuple, _saver):
        return list(obj)

    def new(self, encoded_saved_state):
        return self.TYPE(*encoded_saved_state)


class TensorProductHelper(CodeGenMixinHelper):
    TYPE = o3.TensorProduct
    TYPE_ID = uuid.UUID("1a9a6154-ba68-476b-bb09-2ace5fda5f45")

    # Helpers to make finding easier
    irreps_in1 = mincepy.field("irreps_in1")
    irreps_in2 = mincepy.field("irreps_in2")
    irreps_out = mincepy.field("irreps_out")


class TensorSquareHelper(CodeGenMixinHelper):
    TYPE = o3.TensorSquare
    TYPE_ID = uuid.UUID("aa68e071-92ee-48ad-8909-80f9069f629c")

    # Helpers to make finding easier
    irreps_in = mincepy.field("irreps_in")


class ReducedTensorProductsHelper(CodeGenMixinHelper):
    TYPE = o3.ReducedTensorProducts
    TYPE_ID = uuid.UUID("b87bf7b1-eb3d-4d5d-b65a-3a6780d725a3")

    # Helpers to make finding easier
    irreps_in = mincepy.field("irreps_in")
    irreps_out = mincepy.field("irreps_out")


# endregion

# region nn
class GateHelper(pytorch_types.ModuleHelperStub):
    TYPE = nn.Gate
    TYPE_ID = uuid.UUID("ad05bad7-aa30-4e81-8f61-308640db1878")

    irreps_in = mincepy.field("irreps_in")
    irreps_out = mincepy.field("irreps_out")


class ShortcutHelper(pytorch_types.ModuleHelperStub):
    TYPE = nn._gate._Sortcut
    TYPE_ID = uuid.UUID("c1de96e4-0963-4528-8d8e-87711049cc7b")


class ExtractHelper(CodeGenMixinHelper):
    TYPE = nn.Extract
    TYPE_ID = uuid.UUID("5b2e108e-744a-4537-909b-31826ca6788a")


class ActivationHelper(pytorch_types.ModuleHelperStub):
    TYPE = nn.Activation
    TYPE_ID = uuid.UUID("ee7b6794-d76f-4ac3-bd44-406bc8a9aa1b")


# endregion

# region math
class Normalize2MomHelper(pytorch_types.ModuleHelperStub):
    """We need a little custom saving code here because normalize2mom contains a reference to a function (self.f) which
    mincepy cannot serialise correctly."""

    TYPE = math.normalize2mom
    TYPE_ID = uuid.UUID("fd535825-95ee-42e2-9ce6-0e7a6f43a0ba")

    def yield_hashables(self, module: torch.nn.Module, hasher):
        yield from hasher.yield_hashables(self.cleanup_state(module.__dict__))

    def save_instance_state(self, model: torch.nn.Module, _saver):
        return self.cleanup_state(super().save_instance_state(model, _saver))

    def load_instance_state(
        self, module: torch.nn.Module, saved_state: dict, _referencer
    ):
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
    # nn
    GateHelper,
    ShortcutHelper,
    ExtractHelper,
    ActivationHelper,
    # math
    Normalize2MomHelper,
)
