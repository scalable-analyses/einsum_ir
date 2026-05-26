"""Walker stress tests: deep schedules must not exhaust the call stack."""

from __future__ import annotations

import numpy as np
import pytest

import etops
from etops.ir import TeirBuilder


def _build_deep_chain(depth: int, dtype: str = "f32") -> etops.Teir:
    b = TeirBuilder()
    b.add_tensor("in0", dtype=dtype)
    b.add_tensor("out", dtype=dtype)
    b.add_primitive(
        "cp",
        operation="Copy",
        axes={"M": [], "N": []},
        metadata={"data_type": dtype},
    )
    for i in range(depth):
        b.add_axis(f"a{i}", extent=1, strides_by_tensor={"in0": 0, "out": 0})
    inv = b.add_invocation("inv", primitive="cp")
    cursor = inv
    for i in range(depth - 1, -1, -1):
        cursor = b.add_iteration(f"it{i}", axis=f"a{i}", children=[cursor])
    b.set_roots([cursor])
    return b.finish(validate=True)


@pytest.mark.parametrize("depth", [256, 1024, 5000])
def test_numpy_walker_handles_deep_chain(depth: int) -> None:
    teir = _build_deep_chain(depth)
    op = etops.compile(teir, backend="numpy", optimize=False)
    in0 = np.array([7.0], dtype=np.float32)
    out = np.array([0.0], dtype=np.float32)
    op.execute(in0, out)
    assert out[0] == 7.0


@pytest.mark.tpp
def test_tpp_walker_handles_deep_chain() -> None:
    pytest.importorskip("etops._native")
    teir = _build_deep_chain(100)
    op = etops.compile(teir, backend="tpp", optimize=False)
    in0 = np.array([3.5], dtype=np.float32)
    out = np.array([0.0], dtype=np.float32)
    op.execute(in0, out)
    assert out[0] == 3.5


def test_cycle_in_schedule_rejected() -> None:
    from etops.ir import (
        Axis,
        InvocationNode,
        IterationNode,
        Primitive,
        Schedule,
        Teir,
        Tensor,
    )
    from etops.ir.dtypes import get_dtype

    teir = Teir(
        tensors={"t": Tensor(id="t", dtype=get_dtype("f32"))},
        axes={"a": Axis(id="a", extent=4, strides={"t": 4}, offsets={})},
        primitives={
            "cp": Primitive(
                id="cp",
                operation="Copy",
                axes={"M": (), "N": ()},
                metadata={"data_type": "f32"},
            )
        },
        schedule=Schedule(
            roots=("inv0",),
            iterations={
                "it1": IterationNode(
                    id="it1", axis="a", policy="sequential", children=("it2",)
                ),
                "it2": IterationNode(
                    id="it2", axis="a", policy="sequential", children=("it1",)
                ),
            },
            invocations={"inv0": InvocationNode(id="inv0", primitive="cp")},
        ),
    )
    with pytest.raises(etops.TeirValidationError) as exc_info:
        etops.validate(teir)
    assert any("cycle" in d.message.lower() for d in exc_info.value.diagnostics)


def test_multi_root_forest_executes() -> None:
    b = TeirBuilder()
    b.add_tensor("in0", dtype="f32")
    b.add_tensor("out", dtype="f32")
    b.add_axis("a", extent=4, strides_by_tensor={"in0": 4, "out": 4})
    b.add_primitive(
        "cp", operation="Copy", axes={"M": [], "N": []}, metadata={"data_type": "f32"}
    )
    inv1 = b.add_invocation("inv1", primitive="cp")
    inv2 = b.add_invocation("inv2", primitive="cp")
    root1 = b.add_iteration("it1", axis="a", children=[inv1])
    root2 = b.add_iteration("it2", axis="a", children=[inv2])
    b.set_roots([root1, root2])
    teir = b.finish(validate=True)

    op = etops.compile(teir, backend="numpy", optimize=False)
    in0 = np.arange(4, dtype=np.float32)
    out = np.zeros(4, dtype=np.float32)
    op.execute(in0, out)
    assert np.allclose(out, in0)
