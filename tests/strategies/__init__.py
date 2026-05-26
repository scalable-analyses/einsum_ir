"""Hypothesis strategies for `Teir` and supporting types."""

from __future__ import annotations

import string
from collections.abc import Iterable, Mapping

from hypothesis import assume
from hypothesis import strategies as st

from etops.emit import einsum
from etops.ir import Teir, TeirBuilder
from etops.ir.dtypes import get_dtype

__all__ = [
    "binary_einsum_teir",
    "small_dim_sizes",
    "unary_einsum_teir",
    "unary_relu_teir",
]


def _axis_letter() -> st.SearchStrategy[str]:
    return st.sampled_from(list(string.ascii_lowercase[:8]))


def small_dim_sizes(min_extent: int = 1, max_extent: int = 8) -> st.SearchStrategy[int]:
    """Extent values that keep test runtime small."""

    return st.integers(min_value=min_extent, max_value=max_extent)


_DTYPE_CHOICES: tuple[str, ...] = ("f32", "f64")


@st.composite
def unary_einsum_teir(
    draw: st.DrawFn, *, max_rank: int = 4, dtype: str | None = None
) -> Teir:
    """Generate a unary permutation einsum and emit it.

    Returns a `Teir` produced by `etops.emit.einsum` with a random
    permutation of axes between input and output. When ``dtype`` is None
    the test draws a dtype from ``_DTYPE_CHOICES``.
    """

    rank = draw(st.integers(min_value=1, max_value=max_rank))
    axes = draw(st.lists(_axis_letter(), min_size=rank, max_size=rank, unique=True))
    perm = draw(st.permutations(axes))
    dim_sizes = {ax: draw(small_dim_sizes()) for ax in axes}
    expr = "".join(axes) + "->" + "".join(perm)
    chosen_dtype = dtype if dtype is not None else draw(st.sampled_from(_DTYPE_CHOICES))
    return einsum(expr, dim_sizes=dim_sizes, dtype=chosen_dtype)


@st.composite
def binary_einsum_teir(draw: st.DrawFn, *, max_rank: int = 3) -> Teir:
    """Generate a small binary contraction einsum and emit it.

    Each axis is independently assigned to one of (C, M, N, K). The role
    determines which operands the axis appears in. Within each operand
    the role-axis ordering is randomly permuted so the resulting IR
    covers the full (view, trans_a, trans_b) layout matrix the lowering
    branches on, not just the canonical C+M+K / C+K+N / C+M+N layout.
    """

    # Pick small role cardinalities. K is always at least 1 so the
    # expression remains a non-trivial contraction.
    cm = draw(st.integers(min_value=0, max_value=max_rank))
    cn = draw(st.integers(min_value=0, max_value=max_rank))
    ck = draw(st.integers(min_value=1, max_value=max_rank))
    cb = draw(st.integers(min_value=0, max_value=max_rank))
    total = cm + cn + ck + cb
    assume(total >= 1)
    letters = list(string.ascii_lowercase)
    axes = letters[:total]

    idx = 0
    role_axes: dict[str, list[str]] = {"M": [], "N": [], "K": [], "C": []}
    for role, count in (("M", cm), ("N", cn), ("K", ck), ("C", cb)):
        role_axes[role] = axes[idx : idx + count]
        idx += count

    in0_axes = draw(st.permutations(role_axes["C"] + role_axes["M"] + role_axes["K"]))
    in1_axes = draw(st.permutations(role_axes["C"] + role_axes["K"] + role_axes["N"]))
    out_axes = draw(st.permutations(role_axes["C"] + role_axes["M"] + role_axes["N"]))
    if not in0_axes:
        in0_axes = role_axes["K"][:1]
    if not in1_axes:
        in1_axes = role_axes["K"][:1]
    if not out_axes:
        out_axes = role_axes["M"][:1] or role_axes["C"][:1] or role_axes["K"][:1]

    expr = "".join(in0_axes) + "," + "".join(in1_axes) + "->" + "".join(out_axes)
    dim_sizes = {ax: draw(small_dim_sizes(min_extent=2, max_extent=6)) for ax in axes}
    chosen_dtype = draw(st.sampled_from(_DTYPE_CHOICES))
    return einsum(expr, dim_sizes=dim_sizes, dtype=chosen_dtype)


def _row_major_strides(
    elem_bytes: int, outer_to_inner: Iterable[str], extents: Mapping[str, int]
) -> dict[str, int]:
    """Byte strides for axes laid out row-major from outer to inner."""

    strides: dict[str, int] = {}
    running = elem_bytes
    for axis in reversed(list(outer_to_inner)):
        strides[axis] = running
        running *= extents[axis]
    return strides


@st.composite
def unary_relu_teir(
    draw: st.DrawFn, *, max_rank: int = 4, dtype: str | None = None
) -> Teir:
    """Generate a unary ReLU IR with random rank, extents, dtype, and axis order.

    Builds a single-primitive ReLU IR directly via `TeirBuilder`. Both
    tensors share the same row-major byte strides under a randomly chosen
    axis order; the schedule iterates outer-to-inner around a scalar ReLU
    invocation.

    Constraints:
    - Matching in/out strides only. The TPP backend explicitly rejects
      transposed in/out layouts (libxsmm has no fused transpose-ReLU
      kernel), so the cross-backend property only holds for matching
      layouts.
    - Extents start at 2 to avoid degenerate trivial axes the
      optimization pipeline may collapse.
    """

    rank = draw(st.integers(min_value=1, max_value=max_rank))
    axes = draw(st.lists(_axis_letter(), min_size=rank, max_size=rank, unique=True))
    axis_order = draw(st.permutations(axes))
    dim_sizes = {ax: draw(small_dim_sizes(min_extent=2, max_extent=6)) for ax in axes}
    chosen_dtype = dtype if dtype is not None else draw(st.sampled_from(_DTYPE_CHOICES))
    elem = get_dtype(chosen_dtype).bytes

    strides = _row_major_strides(elem, axis_order, dim_sizes)

    builder = TeirBuilder().set_name("relu")
    builder.add_tensor("in0", dtype=chosen_dtype)
    builder.add_tensor("out", dtype=chosen_dtype)
    for ax in axes:
        builder.add_axis(
            ax,
            extent=dim_sizes[ax],
            strides_by_tensor={"in0": strides[ax], "out": strides[ax]},
        )
    builder.add_primitive(
        "relu",
        operation="ReLU",
        axes={"M": [], "N": []},
        metadata={"data_type": chosen_dtype},
    )
    inv = builder.add_invocation("inv", primitive="relu")
    prev = inv
    # Iterate outer-to-inner matching the row-major layout so the
    # innermost iteration walks the unit-stride axis.
    for ax in reversed(axis_order):
        prev = builder.add_iteration(f"iter_{ax}", axis=ax, children=[prev])
    builder.set_roots([prev])
    return builder.finish(validate=True)
