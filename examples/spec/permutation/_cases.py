"""Programmatic spec entries for permutation regressions.

Each :class:`Case` pairs an einsum-style declaration with a builder that
returns a freshly-constructed :class:`~etops.ir.Teir`. Dict keys follow
the same ``<einsum_family>/<descriptor>`` convention as the textual
``.teir`` artifacts in this directory; the IR ``@name`` attribute is
derived mechanically by replacing the slash with an underscore so the
IR identifier is a single token. The five entries mirror the unary-copy
regressions from the prototype's ``UnaryTpp`` test suite: four are plain
einsum permutations that round-trip through :func:`etops.emit.einsum`;
the strided-input variant uses :class:`~etops.ir.TeirBuilder` directly
so the input view can carry a non-contiguous stride pattern that the
einsum emitter cannot express.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass

from etops.emit import einsum
from etops.ir import Teir, TeirBuilder

__all__ = ["CASES", "Case"]


@dataclass(frozen=True)
class Case:
    """One spec entry: a (re-)derivable IR plus its declared shape.

    ``skip_backends`` enumerates backend names for which the case is known
    not to lower (e.g. backends whose Copy dispatcher requires a
    unit-stride innermost axis on every operand). The spec test honors
    this list and reports a skip with a clear reason.
    """

    expr: str
    extents: Mapping[str, int]
    dtype: str
    build: Callable[[], Teir]
    skip_backends: tuple[str, ...] = ()
    skip_reason: str = ""


def _ir_name(key: str) -> str:
    """Mechanical IR ``@name`` derived from the artifact key.

    The artifact key uses ``<family>/<descriptor>``; the IR identifier
    must match ``[A-Za-z_][A-Za-z0-9_.\\-]*`` (no slash), so we collapse
    the separator into an underscore.
    """

    return key.replace("/", "_")


def _einsum_case(key: str, expr: str, extents: Mapping[str, int], dtype: str) -> Case:
    return Case(
        expr=expr,
        extents=extents,
        dtype=dtype,
        build=lambda: einsum(expr, dim_sizes=extents, dtype=dtype, name=_ir_name(key)),
    )


def _build_abc_bca_strided_input() -> Teir:
    """The input is a 3-D slice (trailing-axis index 0) of a row-major
    ``(3, 5, 4, 7)`` buffer, giving element strides ``(140, 28, 7)`` for
    axes ``(a, b, c)``; the output is row-major contiguous in ``[b, c, a]``
    order.
    """

    elem_bytes = 4  # f32
    b = TeirBuilder().set_name("abc_bca_strided_input")
    b.add_tensor("in0", dtype="f32")
    b.add_tensor("out", dtype="f32")
    b.add_axis(
        "a",
        extent=3,
        strides_by_tensor={"in0": 140 * elem_bytes, "out": elem_bytes},
    )
    b.add_axis(
        "b",
        extent=5,
        strides_by_tensor={
            "in0": 28 * elem_bytes,
            "out": 4 * 3 * elem_bytes,
        },
    )
    b.add_axis(
        "c",
        extent=4,
        strides_by_tensor={
            "in0": 7 * elem_bytes,
            "out": 3 * elem_bytes,
        },
    )
    b.add_primitive(
        "copy",
        operation="Copy",
        axes={"M": [], "N": []},
        metadata={"data_type": "f32"},
    )
    inv = b.add_invocation("i0", primitive="copy")
    n2 = b.add_iteration("c2", axis="c", children=[inv])
    n1 = b.add_iteration("c1", axis="b", children=[n2])
    n0 = b.add_iteration("c0", axis="a", children=[n1])
    b.set_roots([n0])
    return b.finish(validate=True)


CASES: dict[str, Case] = {
    "a_a/scalar_f64": _einsum_case(
        "a_a/scalar_f64",
        "a->a",
        extents={"a": 3},
        dtype="f64",
    ),
    "abc_cba/scalar_f64": _einsum_case(
        "abc_cba/scalar_f64",
        "[a,b,c]->[c,b,a]",
        extents={"a": 3, "b": 5, "c": 4},
        dtype="f64",
    ),
    "rank9_arbitrary/scalar": _einsum_case(
        "rank9_arbitrary/scalar",
        "[d0,d1,d2,d3,d4,d5,d6,d7,d8]->[d2,d1,d4,d0,d5,d7,d3,d8,d6]",
        extents={
            "d0": 3,
            "d1": 5,
            "d2": 4,
            "d3": 7,
            "d4": 2,
            "d5": 5,
            "d6": 3,
            "d7": 8,
            "d8": 6,
        },
        dtype="f32",
    ),
    "rank9_same_inner/scalar": _einsum_case(
        "rank9_same_inner/scalar",
        "[d0,d1,d2,d3,d4,d5,d6,d7,d8]->[d2,d1,d4,d0,d5,d7,d3,d6,d8]",
        extents={
            "d0": 3,
            "d1": 5,
            "d2": 4,
            "d3": 7,
            "d4": 2,
            "d5": 5,
            "d6": 3,
            "d7": 8,
            "d8": 6,
        },
        dtype="f32",
    ),
    "abc_bca/strided_input": Case(
        expr="[a,b,c]->[b,c,a]",
        extents={"a": 3, "b": 5, "c": 4},
        dtype="f32",
        build=_build_abc_bca_strided_input,
        skip_backends=("tpp",),
        skip_reason=(
            "TPP Copy dispatcher requires a unit-stride innermost axis on"
            " every operand; this case's input view has no unit-stride axis"
        ),
    ),
}
