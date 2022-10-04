# -*- coding: utf-8 -*-
"""Module that provides interoperability between pymatgen and mincepy"""

import uuid

from e3nn import o3
import mincepy


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


class InstructionHelper(mincepy.common_helpers.TupleHelper):
    TYPE = o3.Instruction
    TYPE_ID = uuid.UUID("a76e6544-39aa-4291-89ab-aa7c48c97484")

    # pylint: disable=arguments-differ

    def save_instance_state(self, obj: tuple, _saver):
        return list(obj)

    def new(self, encoded_saved_state):
        return self.TYPE(*encoded_saved_state)


class TensorProduct(CodeGenMixinHelper):
    TYPE = o3.TensorProduct
    TYPE_ID = uuid.UUID("1a9a6154-ba68-476b-bb09-2ace5fda5f45")

    # Helpers to make finding easier
    irreps_in1 = mincepy.field("irreps_in1")
    irreps_in2 = mincepy.field("irreps_in2")
    irreps_out = mincepy.field("irreps_out")


class ReducedTensorProductsHelper(CodeGenMixinHelper):
    TYPE = o3.ReducedTensorProducts
    TYPE_ID = uuid.UUID("b87bf7b1-eb3d-4d5d-b65a-3a6780d725a3")

    # Helpers to make finding easier
    irreps_in = mincepy.field("irreps_in")
    irreps_out = mincepy.field("irreps_out")


TYPES = (
    IrrepHelper,
    IrrepsHelper,
    InstructionHelper,
    TensorProduct,
    ReducedTensorProductsHelper,
)
