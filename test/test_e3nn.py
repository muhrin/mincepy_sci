# -*- coding: utf-8 -*-
# pylint: disable=wrong-import-position
import random

import pytest

torch = pytest.importorskip("torch")
import mincepy
from mincepy import testing
from e3nn import nn
from e3nn import o3
import e3nn.util.test


# region: From e3nn tests.  Copyright belongs to e3nn under MIT license

# pylint: disable=invalid-name


def make_tp(
    l1,
    p1,
    l2,
    p2,
    lo,
    po,
    mode,
    weight,
    mul: int = 25,
    path_weights: bool = True,
    **kwargs
):
    def mul_out(mul):
        if mode == "uvuv":
            return mul**2
        if mode == "uvu<v":
            return mul * (mul - 1) // 2
        return mul

    try:
        # Need to set the seed so that we can get a deterministic TP
        torch.manual_seed(0xDEADBEEF)
        return o3.TensorProduct(
            [(mul, (l1, p1)), (19, (l1, p1))],
            [(mul, (l2, p2)), (19, (l2, p2))],
            [(mul_out(mul), (lo, po)), (mul_out(19), (lo, po))],
            [
                (0, 0, 0, mode, weight),
                (1, 1, 1, mode, weight),
                (0, 0, 1, "uvw", True, 0.5 if path_weights else 1.0),
                (0, 1, 1, "uvw", True, 0.2 if path_weights else 1.0),
            ],
            compile_left_right=True,
            compile_right=True,
            **kwargs,
        )
    except AssertionError:
        return None


def random_params(n=25):
    params = set()
    while len(params) < n:
        l1 = random.randint(0, 2)
        p1 = random.choice([-1, 1])
        l2 = random.randint(0, 2)
        p2 = random.choice([-1, 1])
        lo = random.randint(0, 2)
        po = random.choice([-1, 1])
        mode = random.choice(["uvw", "uvu", "uvv", "uuw", "uuu", "uvuv"])
        weight = random.choice([True, False])
        if make_tp(l1, p1, l2, p2, lo, po, mode, weight) is not None:
            params.add((l1, p1, l2, p2, lo, po, mode, weight))
    return params


# endregion

# region o3
def test_saving_irrep(historian: mincepy.Historian):
    irrep_str = "4e"
    loaded = testing.do_round_trip(
        historian,
        lambda: mincepy.builtins.ObjProxy(o3.Irrep(irrep_str)),
    )()

    assert historian.hash(loaded) is not None
    assert str(loaded) == irrep_str


def test_saving_irreps(historian: mincepy.Historian):
    irreps_str = "2x0e+3x1o"
    loaded = testing.do_round_trip(
        historian,
        lambda: mincepy.builtins.ObjProxy(o3.Irreps(irreps_str)),
    )()

    assert historian.hash(loaded) is not None
    assert str(loaded) == irreps_str


@pytest.mark.parametrize("l1, p1, l2, p2, lo, po, mode, weight", random_params(n=1))
def test_saving_tensor_product(
    l1, p1, l2, p2, lo, po, mode, weight, historian: mincepy.Historian
):
    args = l1, p1, l2, p2, lo, po, mode, weight
    tp = make_tp(*args)
    # Saved TP
    tp2 = testing.do_round_trip(historian, make_tp, *args)
    # JITed, saved TP
    tp3 = testing.do_round_trip(historian, _make_auto_jitable, *args)

    x1 = torch.randn(2, tp.irreps_in1.dim)
    x2 = torch.randn(2, tp.irreps_in2.dim)
    res1 = tp(x1, x2)
    res2 = tp2(x1, x2)
    res3 = tp3(x1, x2)
    assert torch.allclose(res1, res2)
    assert torch.allclose(res1, res3)


def test_saving_tensor_square(historian: mincepy.Historian):
    irreps = "1o"
    tp = o3.TensorSquare(irreps)

    loaded = testing.do_round_trip(historian, o3.TensorSquare, irreps)
    assert loaded.irreps_in == tp.irreps_in


def _make_auto_jitable(*args, **kwargs):
    tp = make_tp(*args, **kwargs)
    e3nn.util.test.assert_auto_jitable(tp)
    return tp


def test_saving_reduced_tensor_product(historian: mincepy.Historian):
    perm = "ij=-ji"
    irreps = o3.Irreps("5x0e + 1e")
    kwargs = dict(i=irreps)

    reference = o3.ReducedTensorProducts(perm, **kwargs)
    loaded: o3.ReducedTensorProducts = testing.do_round_trip(
        historian, o3.ReducedTensorProducts, perm, **kwargs
    )

    assert reference.irreps_in == loaded.irreps_in
    assert reference.irreps_out == loaded.irreps_out

    sph = irreps.randn(1, -1)
    # Check that the result of the tensor product is the same before and after
    assert torch.all(reference(sph, sph) == loaded(sph, sph))


# endregion

# region nn


def test_gate_save_load(historian: mincepy.Historian):
    irreps_scalars, act_scalars, irreps_gates, act_gates, irreps_gated = (
        o3.Irreps("16x0o"),
        [torch.tanh],
        o3.Irreps("32x0o"),
        [torch.tanh],
        o3.Irreps("16x1e+16x1o"),
    )
    gate = nn.Gate(irreps_scalars, act_scalars, irreps_gates, act_gates, irreps_gated)

    loaded = testing.do_round_trip(
        historian,
        nn.Gate,
        irreps_scalars,
        act_scalars,
        irreps_gates,
        act_gates,
        irreps_gated,
    )

    assert loaded.irreps_in == gate.irreps_in
    assert loaded.irreps_out == gate.irreps_out


# endregion
