"""Builders and locator constants for TEIR test fixtures.

The seven spec-aligned schedules (permutation x {scalar, tiled};
batched GEMM x {scalar, reordered}; tensor contraction x {scalar, gemm,
brgemm}) live as canonical textual IR under ``examples/spec/``. The
``SPEC_TEIR_FILES`` mapping below is the single source of truth for
their paths; tests load each via ``etops.textir.parse((path).read_text())``.

The remaining builders cover schedules without a ``.teir`` counterpart:

- ``build_gemm_with_layout`` / ``build_brgemm_with_layout`` parametrize
  the unit-stride axis per tensor, spanning the eight corners of the
  (view x trans_a x trans_b) dispatch matrix the TPP / BLAS lowerings
  branch on.
- ``build_hadamard_contraction`` / ``build_outer_product_contraction``
  exercise the placeholder axes ``EnsureKernelShape`` synthesizes for
  pure-C and zero-K einsum shapes.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Literal

from etops.ir import Teir, TeirBuilder
from etops.ir.dtypes import get_dtype

__all__ = [
    "BRGEMM_LAYOUT_SIZES",
    "GEMM_LAYOUT_SIZES",
    "SPEC_TEIR_FILES",
    "build_brgemm_with_layout",
    "build_gemm_with_layout",
    "build_hadamard_contraction",
    "build_outer_product_contraction",
    "gemm_layout_einsum_expr",
    "gemm_layout_shapes",
]


_SPEC_ROOT: Path = Path(__file__).resolve().parents[2] / "examples" / "spec"

SPEC_TEIR_FILES: Mapping[str, Path] = {
    "scalar_permutation": _SPEC_ROOT / "permutation" / "scalar.teir",
    "tiled_permutation": _SPEC_ROOT / "permutation" / "tiled.teir",
    "scalar_bgemm": _SPEC_ROOT / "batched_gemm" / "scalar.teir",
    "reordered_scalar_bgemm": _SPEC_ROOT / "batched_gemm" / "scalar_reordered.teir",
    "scalar_tensor_contraction": _SPEC_ROOT / "tensor_contraction" / "scalar.teir",
    "gemm_tensor_contraction": _SPEC_ROOT / "tensor_contraction" / "gemm.teir",
    "brgemm_tensor_contraction": _SPEC_ROOT / "tensor_contraction" / "brgemm.teir",
}


GEMM_LAYOUT_SIZES: dict[str, int] = {"m": 4, "n": 6, "k": 8}
BRGEMM_LAYOUT_SIZES: dict[str, int] = {"m": 4, "n": 6, "k": 8, "kb": 3}


def _row_major_strides(
    elem_bytes: int,
    inner_to_outer: tuple[str, ...],
    extents: Mapping[str, int],
) -> dict[str, int]:
    """Byte strides for axes laid out row-major from inner to outer."""

    strides: dict[str, int] = {}
    running = elem_bytes
    for axis in inner_to_outer:
        strides[axis] = running
        running *= extents[axis]
    return strides


def _merge_tensor_strides(
    b: TeirBuilder,
    axis_order: tuple[str, ...],
    extents: Mapping[str, int],
    per_tensor: Mapping[str, Mapping[str, int]],
) -> None:
    """Add axes whose strides are sparse across multiple tensors."""

    for ax in axis_order:
        strides: dict[str, int] = {}
        for tid, table in per_tensor.items():
            if ax in table:
                strides[tid] = table[ax]
        b.add_axis(ax, extent=extents[ax], strides_by_tensor=strides)


# --------------------------------------------------------------------------
# Layout-parametrized GEMM / BRGEMM
#
# The three ``*_unit`` knobs each pick which of an operand's two role axes
# has unit byte stride. The eight combinations exhaustively cover the
# (view x trans_a x trans_b) corners that ``plan_libxsmm_contraction`` /
# the BLAS dispatcher branch on. Mapping:
#
#   out_unit == "M" → direct view (libxsmm A = in0, B = in1).
#       trans_a iff in0_unit == "K"
#       trans_b iff in1_unit == "N"
#   out_unit == "N" → swap view   (libxsmm A = in1, B = in0).
#       trans_a iff in1_unit == "K"
#       trans_b iff in0_unit == "M"
#
# --------------------------------------------------------------------------


def _gemm_layout_strides(
    in0_unit: Literal["M", "K"],
    in1_unit: Literal["K", "N"],
    out_unit: Literal["M", "N"],
    sizes: Mapping[str, int],
    elem: int,
) -> tuple[dict[str, int], dict[str, int], dict[str, int]]:
    m, n, k = sizes["m"], sizes["n"], sizes["k"]
    in0 = {"m": elem, "k": m * elem} if in0_unit == "M" else {"k": elem, "m": k * elem}
    in1 = {"k": elem, "n": k * elem} if in1_unit == "K" else {"n": elem, "k": n * elem}
    out = {"m": elem, "n": m * elem} if out_unit == "M" else {"n": elem, "m": n * elem}
    return in0, in1, out


def build_gemm_with_layout(
    *,
    in0_unit: Literal["M", "K"] = "K",
    in1_unit: Literal["K", "N"] = "N",
    out_unit: Literal["M", "N"] = "N",
    sizes: Mapping[str, int] | None = None,
) -> Teir:
    """Single-invocation f32 GEMM IR with caller-controlled unit-stride per tensor.

    Layout knobs:

    * ``in0_unit``: which axis (``M`` or ``K``) is unit-stride on ``in0``.
    * ``in1_unit``: which axis (``K`` or ``N``) is unit-stride on ``in1``.
    * ``out_unit``: which axis (``M`` or ``N``) is unit-stride on ``out``.

    Caller is responsible for zeroing ``out`` before execution.
    """

    sizes = dict(sizes or GEMM_LAYOUT_SIZES)
    elem = get_dtype("f32").bytes
    in0, in1, out = _gemm_layout_strides(in0_unit, in1_unit, out_unit, sizes, elem)

    builder = TeirBuilder().set_name("gemm_layout")
    builder.add_tensor("in0", dtype="f32")
    builder.add_tensor("in1", dtype="f32")
    builder.add_tensor("out", dtype="f32")
    _merge_tensor_strides(
        builder, ("m", "n", "k"), sizes, {"in0": in0, "in1": in1, "out": out}
    )
    builder.add_primitive(
        "contraction",
        operation="Contraction",
        axes={"M": ["m"], "N": ["n"], "K": ["k"]},
        metadata={"data_type": "f32"},
    )
    inv = builder.add_invocation("inv", primitive="contraction")
    builder.set_roots([inv])
    return builder.finish(validate=True)


def build_brgemm_with_layout(
    *,
    in0_unit: Literal["M", "K"] = "K",
    in1_unit: Literal["K", "N"] = "N",
    out_unit: Literal["M", "N"] = "N",
    sizes: Mapping[str, int] | None = None,
) -> Teir:
    """Single-invocation f32 BRGEMM IR with caller-controlled inner-K unit-stride.

    BRGEMM convention: the K role has two axes ``[kb, k]`` where ``kb`` is
    the outer batch-reduce dimension and ``k`` is the inner GEMM-K. The
    layout knobs control unit-stride on the **inner** K axis; ``kb`` is
    placed at the outermost slot so its stride is the product of inner
    strides — exactly the BR-stride libxsmm expects.

    Caller is responsible for zeroing ``out`` before execution.
    """

    sizes = dict(sizes or BRGEMM_LAYOUT_SIZES)
    m, n, k = sizes["m"], sizes["n"], sizes["k"]
    elem = get_dtype("f32").bytes
    in0_inner, in1_inner, out = _gemm_layout_strides(
        in0_unit, in1_unit, out_unit, sizes, elem
    )

    builder = TeirBuilder().set_name("brgemm_layout")
    builder.add_tensor("in0", dtype="f32")
    builder.add_tensor("in1", dtype="f32")
    builder.add_tensor("out", dtype="f32")
    _merge_tensor_strides(
        builder,
        ("m", "n", "k"),
        sizes,
        {"in0": in0_inner, "in1": in1_inner, "out": out},
    )
    builder.add_axis(
        "kb",
        extent=sizes["kb"],
        strides_by_tensor={"in0": m * k * elem, "in1": k * n * elem},
    )
    builder.add_primitive(
        "contraction",
        operation="Contraction",
        axes={"M": ["m"], "N": ["n"], "K": ["kb", "k"]},
        metadata={"data_type": "f32"},
    )
    inv = builder.add_invocation("inv", primitive="contraction")
    builder.set_roots([inv])
    return builder.finish(validate=True)


def gemm_layout_shapes(
    in0_unit: str, in1_unit: str, out_unit: str, sizes: Mapping[str, int]
) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int]]:
    """Per-tensor ndarray shape implied by the unit-stride layout knobs."""

    m, n, k = sizes["m"], sizes["n"], sizes["k"]
    in0 = (k, m) if in0_unit == "M" else (m, k)
    in1 = (n, k) if in1_unit == "K" else (k, n)
    out = (n, m) if out_unit == "M" else (m, n)
    return in0, in1, out


def gemm_layout_einsum_expr(in0_unit: str, in1_unit: str, out_unit: str) -> str:
    """Einsum expression matching the shapes produced by :func:`gemm_layout_shapes`."""

    in0 = "km" if in0_unit == "M" else "mk"
    in1 = "nk" if in1_unit == "K" else "kn"
    out = "nm" if out_unit == "M" else "mn"
    return f"{in0},{in1}->{out}"


# --------------------------------------------------------------------------
# Pure-C Hadamard and outer product
#
# These two einsum shapes are the canonical test cases for the placeholder
# axes ``EnsureKernelShape`` synthesizes when a Contraction's einsum lacks
# one or more role categories. Hadamard has *no* M / N / K axes; outer
# product has M and N but no K. Both must lower cleanly through the TPP
# and BLAS dispatchers after optimization.
# --------------------------------------------------------------------------


def build_hadamard_contraction(
    sizes: Mapping[str, int] | None = None,
    dtype: str = "f32",
) -> Teir:
    """Build the pure-C Hadamard schedule ``abc,abc->abc``.

    All three axes are batch (``C``) — they appear on every tensor — so
    the einsum carries no M / N / K. The schedule iterates ``a, b, c``
    over the contraction site and emits a sibling ``Zero`` that
    initializes each output element before the multiply-accumulate.
    """

    sizes = dict(sizes or {"a": 2, "b": 3, "c": 4})
    elem = get_dtype(dtype).bytes
    shared = _row_major_strides(elem, ("c", "b", "a"), sizes)

    builder = TeirBuilder().set_name("hadamard")
    builder.add_tensor("in0", dtype=dtype)
    builder.add_tensor("in1", dtype=dtype)
    builder.add_tensor("out", dtype=dtype)
    for ax in ("a", "b", "c"):
        builder.add_axis(
            ax,
            extent=sizes[ax],
            strides_by_tensor={"in0": shared[ax], "in1": shared[ax], "out": shared[ax]},
        )
    builder.add_primitive(
        "zero",
        operation="Zero",
        axes={"M": [], "N": []},
        metadata={"data_type": dtype},
    )
    builder.add_primitive(
        "contraction",
        operation="Contraction",
        axes={"M": [], "N": [], "K": []},
        metadata={"data_type": dtype},
    )
    inv_zero = builder.add_invocation("inv_zero", primitive="zero")
    inv_ct = builder.add_invocation("inv_contract", primitive="contraction")
    iter_c = builder.add_iteration("iter_c", axis="c", children=[inv_zero, inv_ct])
    iter_b = builder.add_iteration("iter_b", axis="b", children=[iter_c])
    iter_a = builder.add_iteration("iter_a", axis="a", children=[iter_b])
    builder.set_roots([iter_a])
    return builder.finish(validate=True)


def build_outer_product_contraction(
    sizes: Mapping[str, int] | None = None,
    dtype: str = "f32",
) -> Teir:
    """Build the outer-product schedule ``a,b->ab``.

    ``a`` is M (lives on ``in0`` and ``out``); ``b`` is N (lives on
    ``in1`` and ``out``); there is no contraction axis — the K role is
    empty in the emitted IR.
    """

    sizes = dict(sizes or {"a": 3, "b": 4})
    a_size, b_size = sizes["a"], sizes["b"]
    elem = get_dtype(dtype).bytes

    builder = TeirBuilder().set_name("outer_product")
    builder.add_tensor("in0", dtype=dtype)
    builder.add_tensor("in1", dtype=dtype)
    builder.add_tensor("out", dtype=dtype)
    builder.add_axis(
        "a",
        extent=a_size,
        strides_by_tensor={"in0": elem, "out": b_size * elem},
    )
    builder.add_axis(
        "b",
        extent=b_size,
        strides_by_tensor={"in1": elem, "out": elem},
    )
    builder.add_primitive(
        "zero",
        operation="Zero",
        axes={"M": [], "N": []},
        metadata={"data_type": dtype},
    )
    builder.add_primitive(
        "contraction",
        operation="Contraction",
        axes={"M": [], "N": [], "K": []},
        metadata={"data_type": dtype},
    )
    inv_zero = builder.add_invocation("inv_zero", primitive="zero")
    inv_ct = builder.add_invocation("inv_contract", primitive="contraction")
    iter_b = builder.add_iteration("iter_b", axis="b", children=[inv_zero, inv_ct])
    iter_a = builder.add_iteration("iter_a", axis="a", children=[iter_b])
    builder.set_roots([iter_a])
    return builder.finish(validate=True)
