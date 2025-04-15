import uuid

from flax import linen
import mincepy
from typing_extensions import override


class LinenDenseHelper(
    mincepy.TypeHelper,
    obj_type=linen.Dense,
    type_id=uuid.UUID("fb7ba12b-974e-4b52-bc02-8e0d693dd39a"),
    immutable=True,
):

    @override
    def yield_hashables(self, dense: linen.Dense, hasher, /):
        yield dense.features
        yield dense.use_bias
        yield dense.dtype
        yield dense.param_dtype
        yield dense.precision
        yield dense.kernel_init
        yield dense.bias_init

    @override
    def save_instance_state(self, sequential: linen.Sequential, saver: mincepy.Saver, /):
        return saver.save_state(sequential.layers)

    @override
    def new(self, saved_state: list, /) -> linen.Sequential:
        return linen.Sequential(saved_state)

    @override
    def load_instance_state(self, sequential: linen.Sequential, saved_state: list, _referencer, /):
        # Nothing to do, all done in new()
        pass


class LinenDenseGeneralHelper(
    mincepy.TypeHelper,
    obj_type=linen.DenseGeneral,
    type_id=uuid.UUID("2f48a5ee-03d9-4705-89ff-5bcd8d815afe"),
    immutable=True,
):
    pass


class LinenSequentialHelper(
    mincepy.TypeHelper,
    obj_type=linen.Sequential,
    type_id=uuid.UUID("a2ebfb10-5719-4593-b77d-bb4c54092357"),
    immutable=True,
):

    @override
    def yield_hashables(self, sequential: linen.Sequential, hasher, /):
        yield from hasher.yield_hashables(sequential.layers)

    @override
    def save_instance_state(self, sequential: linen.Sequential, saver: mincepy.Saver, /):
        return saver.save_state(sequential.layers)

    @override
    def new(self, saved_state: list, /) -> linen.Sequential:
        return linen.Sequential(saved_state)

    @override
    def load_instance_state(self, sequential: linen.Sequential, saved_state: list, _referencer, /):
        # Nothing to do, all done in new()
        pass
