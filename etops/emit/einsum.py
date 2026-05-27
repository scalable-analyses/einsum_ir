"""Einsum-string emitter.

Parses an explicit einsum string ``"<in0_axes>,<in1_axes>->.<out_axes>"``
(or ``"<in_axes>-><out_axes>"`` for unary) and emits a `Teir` with a
default scalar schedule:

- Outer chain over batch / free axes (in einsum string order).
- For binary contractions: a sibling ``Zero`` invocation initializing
  the output tile, placed just outside the K chain so it executes once
  per output tile; followed by the inner chain over the contracted axes
  ending in a scalar ``Contraction`` invocation.
- For unary operations: a chain over every axis ending in a single ``Copy``
  invocation.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence

from etops.diag import TeirEmissionError
from etops.ir import Teir, TeirBuilder
from etops.ir.dtypes import DataType, get_dtype
from etops.transforms._canonicalize import ROLE_PREFIX

__all__ = ["einsum"]

_DEFAULT_BINARY_NAMES = ("in0", "in1", "out")
_DEFAULT_UNARY_NAMES = ("in0", "out")

_AXIS_NAME_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def einsum(
    expr: str,
    *,
    tensor_names: tuple[str, ...] | None = None,
    dim_sizes: Mapping[str, int],
    dtype: str | DataType = "f32",
    name: str | None = None,
) -> Teir:
    """Emit a `Teir` from an explicit einsum string.

    Args:
        expr: Explicit einsum, e.g. ``"trus,pqtu->pqrs"`` or
            ``"abcd->dcba"``. The output substring is mandatory.
        tensor_names: Names assigned to the tensors. Defaults to
            ``("in0", "in1", "out")`` for binary and ``("in0", "out")``
            for unary.
        dim_sizes: Map of axis name → extent. Every axis appearing in the
            expression must have an entry.
        dtype: Element type used for every tensor.
        name: Optional symbolic name attached to the resulting `Teir`.

    Returns:
        A validated, immutable `Teir`.
    """

    inputs, output = _split_expr(expr)
    if len(inputs) not in (1, 2):
        raise TeirEmissionError(
            f"einsum: v1 supports unary and binary expressions; got {len(inputs)} inputs",
        )

    operand_axes = [*inputs, output]
    if tensor_names is None:
        tensor_names = (
            _DEFAULT_UNARY_NAMES if len(inputs) == 1 else _DEFAULT_BINARY_NAMES
        )
    if len(tensor_names) != len(operand_axes):
        raise TeirEmissionError(
            f"einsum: expected {len(operand_axes)} tensor names; got {len(tensor_names)}",
        )

    # Validate axes referenced.
    all_axes: dict[str, None] = {}
    for axes in operand_axes:
        for ax in axes:
            all_axes[ax] = None
    for ax in all_axes:
        if ax not in dim_sizes:
            raise TeirEmissionError(f"einsum: missing dim_size for axis {ax!r}")

    resolved_dtype = dtype if isinstance(dtype, DataType) else get_dtype(dtype)

    if len(inputs) == 1:
        return _emit_unary(
            in_axes=inputs[0],
            out_axes=output,
            tensor_names=tensor_names,
            dim_sizes=dim_sizes,
            dtype=resolved_dtype,
            name=name,
        )
    return _emit_binary(
        in0_axes=inputs[0],
        in1_axes=inputs[1],
        out_axes=output,
        tensor_names=tensor_names,
        dim_sizes=dim_sizes,
        dtype=resolved_dtype,
        name=name,
    )


# --------------------------------------------------------------------------
# split + classify
# --------------------------------------------------------------------------


def _split_expr(expr: str) -> tuple[tuple[tuple[str, ...], ...], tuple[str, ...]]:
    if "->" not in expr:
        raise TeirEmissionError(
            "einsum: explicit output is required; expression must contain '->'",
        )
    lhs, rhs = expr.split("->", 1)
    lhs = lhs.strip()
    rhs = rhs.strip()
    operands = _split_operands(lhs)
    parsed_inputs = tuple(_split_axes(op) for op in operands)
    out_axes = _split_axes(rhs.strip())
    return parsed_inputs, out_axes


def _split_operands(lhs: str) -> list[str]:
    """Split the left-hand side of an einsum string into per-operand chunks.

    When any operand is bracketed (``[a,b]``) the entire LHS must be a
    comma-separated list of bracketed groups; this disambiguates against the
    bare single-letter form (``ab,cd``). Mixing the two forms is rejected to
    keep the grammar unambiguous.
    """

    if "[" not in lhs and "]" not in lhs:
        return [op.strip() for op in lhs.split(",")]
    # Bracketed form. Operands are top-level `[...]` groups separated by
    # commas; commas inside brackets are axis separators, not operand
    # separators.
    operands: list[str] = []
    i, n = 0, len(lhs)
    while i < n:
        # Skip whitespace and commas between operands.
        while i < n and (lhs[i].isspace() or lhs[i] == ","):
            i += 1
        if i >= n:
            break
        if lhs[i] != "[":
            raise TeirEmissionError(
                f"einsum: expected '[' at offset {i} of {lhs!r};"
                " mixing bracketed and bare operands is not allowed",
            )
        depth = 1
        start = i
        i += 1
        while i < n and depth > 0:
            if lhs[i] == "[":
                depth += 1
            elif lhs[i] == "]":
                depth -= 1
            i += 1
        if depth != 0:
            raise TeirEmissionError(
                f"einsum: unbalanced '[' in {lhs!r}",
            )
        operands.append(lhs[start:i].strip())
    return operands


def _split_axes(s: str) -> tuple[str, ...]:
    """Split an operand into axis names.

    Two accepted forms:

    1. **Bracketed**: ``[a, b, c1]`` — comma-separated identifiers inside
       square brackets. Multi-character names allowed.
    2. **Bare single-letter**: ``abc`` — each character is one axis.

    Whitespace is tolerated. Empty operands yield the empty tuple.
    """

    s = s.strip()
    if not s:
        return ()
    if s.startswith("["):
        if not s.endswith("]"):
            raise TeirEmissionError(
                f"einsum: bracketed operand {s!r} is missing its closing ']'",
            )
        inner = s[1:-1].strip()
        if not inner:
            return ()
        parts = [p.strip() for p in inner.split(",")]
        for p in parts:
            if not _AXIS_NAME_RE.fullmatch(p):
                raise TeirEmissionError(f"einsum: invalid axis name {p!r}")
        return tuple(parts)
    # Bare single-letter form: every character is one axis. Reject runs
    # that contain non-identifier characters; multi-character names
    # require the bracketed form.
    if not s[0].isalpha():
        raise TeirEmissionError(f"einsum: invalid operand {s!r}")
    for ch in s:
        if not ch.isalnum() and ch != "_":
            raise TeirEmissionError(f"einsum: invalid axis character {ch!r} in {s!r}")
    return tuple(s)


# --------------------------------------------------------------------------
# unary
# --------------------------------------------------------------------------


def _emit_unary(
    *,
    in_axes: tuple[str, ...],
    out_axes: tuple[str, ...],
    tensor_names: tuple[str, ...],
    dim_sizes: Mapping[str, int],
    dtype: DataType,
    name: str | None,
) -> Teir:
    if set(in_axes) != set(out_axes):
        raise TeirEmissionError(
            "einsum: unary expression must use the same axis set on both sides;"
            f" got {in_axes!r} -> {out_axes!r}",
        )
    in_name, out_name = tensor_names

    b = TeirBuilder().set_name(name or "")
    b.add_tensor(in_name, dtype=dtype)
    b.add_tensor(out_name, dtype=dtype)

    # Per-tensor element strides (row-major in the order axes appear in the
    # operand's einsum substring), converted to byte strides.
    element_bytes = dtype.bytes
    in_stride = _row_major_strides(in_axes, dim_sizes, element_bytes)
    out_stride = _row_major_strides(out_axes, dim_sizes, element_bytes)

    for ax in in_axes:  # iteration order = input order
        b.add_axis(
            ax,
            extent=dim_sizes[ax],
            strides_by_tensor={
                in_name: in_stride[ax],
                out_name: out_stride[ax],
            },
        )

    b.add_primitive(
        "copy",
        operation="Copy",
        axes={"M": [], "N": []},
        metadata={"data_type": dtype.name},
    )

    # Schedule pre-order: outermost iteration over in_axes[0] down to the
    # innermost iteration over in_axes[-1], then the Copy invocation. Every
    # axis on a 2-tensor IR is classified ``C`` by `dim_roles`, so
    # iteration ids run ``c0, c1, ...`` and the lone invocation is ``i0``.
    iter_ids = [f"c{i}" for i in range(len(in_axes))]
    inv = b.add_invocation("i0", primitive="copy")
    child = inv
    for ax, node_id in zip(reversed(in_axes), reversed(iter_ids), strict=True):
        child = b.add_iteration(node_id, axis=ax, children=[child])
    b.set_roots([child])
    return b.finish(validate=True)


# --------------------------------------------------------------------------
# binary
# --------------------------------------------------------------------------


def _emit_binary(
    *,
    in0_axes: tuple[str, ...],
    in1_axes: tuple[str, ...],
    out_axes: tuple[str, ...],
    tensor_names: tuple[str, ...],
    dim_sizes: Mapping[str, int],
    dtype: DataType,
    name: str | None,
) -> Teir:
    in0_name, in1_name, out_name = tensor_names

    in0_set = set(in0_axes)
    in1_set = set(in1_axes)
    out_set = set(out_axes)
    union = in0_set | in1_set | out_set
    classification: dict[str, str] = {}
    for ax in union:
        in0_has = ax in in0_set
        in1_has = ax in in1_set
        out_has = ax in out_set
        if in0_has and in1_has and out_has:
            classification[ax] = "C"  # batch
        elif in0_has and out_has and not in1_has:
            classification[ax] = "M"
        elif in1_has and out_has and not in0_has:
            classification[ax] = "N"
        elif in0_has and in1_has and not out_has:
            classification[ax] = "K"
        else:
            raise TeirEmissionError(
                f"einsum: axis {ax!r} has unsupported occurrence pattern"
                f" (in0={in0_has}, in1={in1_has}, out={out_has})",
            )

    b = TeirBuilder().set_name(name or "")
    b.add_tensor(in0_name, dtype=dtype)
    b.add_tensor(in1_name, dtype=dtype)
    b.add_tensor(out_name, dtype=dtype)

    element_bytes = dtype.bytes
    in0_stride = _row_major_strides(in0_axes, dim_sizes, element_bytes)
    in1_stride = _row_major_strides(in1_axes, dim_sizes, element_bytes)
    out_stride = _row_major_strides(out_axes, dim_sizes, element_bytes)

    # Determine axis order for the schedule: C, M, N (output order), then
    # K (in0 order).
    c_axes = [ax for ax in out_axes if classification[ax] == "C"]
    m_axes = [ax for ax in out_axes if classification[ax] == "M"]
    n_axes = [ax for ax in out_axes if classification[ax] == "N"]
    k_axes = [ax for ax in in0_axes if classification[ax] == "K"]
    outer_axes = [*c_axes, *m_axes, *n_axes]
    # k_axes determined from in0_axes order; this is the inner chain.

    for ax in (*outer_axes, *k_axes):
        strides: dict[str, int] = {}
        if ax in in0_stride:
            strides[in0_name] = in0_stride[ax]
        if ax in in1_stride:
            strides[in1_name] = in1_stride[ax]
        if ax in out_stride:
            strides[out_name] = out_stride[ax]
        b.add_axis(ax, extent=dim_sizes[ax], strides_by_tensor=strides)

    b.add_primitive(
        "zero",
        operation="Zero",
        axes={"M": [], "N": []},
        metadata={"data_type": dtype.name},
    )
    b.add_primitive(
        "contraction",
        operation="Contraction",
        axes={"M": [], "N": [], "K": []},
        metadata={"data_type": dtype.name},
    )

    outer_ids = _canonical_iter_ids(outer_axes, classification)
    k_ids = [f"k{i}" for i in range(len(k_axes))]

    if not k_axes:
        # Pure outer-product / element-wise binary; no K axes. The Zero is
        # unguarded, the contraction is a single FMA, and both are children
        # of the innermost free-axis iteration. Pre-order visits the outer
        # chain first, then ``inv_zero`` (``i0``), then ``inv_contract``
        # (``i1``).
        inv_zero = b.add_invocation("i0", primitive="zero")
        inv_ct = b.add_invocation("i1", primitive="contraction")
        outer_node: str = ""
        for ax, nid in zip(reversed(outer_axes), reversed(outer_ids), strict=True):
            children_ids = [outer_node] if outer_node else [inv_zero, inv_ct]
            outer_node = b.add_iteration(nid, axis=ax, children=children_ids)
        if not outer_axes:
            # Scalar contraction with no axes — both invocations are roots.
            b.set_roots([inv_zero, inv_ct])
        else:
            b.set_roots([outer_node])
        return b.finish(validate=True)

    # Default emitter chooses the "reordered" schedule shape (free axes
    # outermost, K loop innermost). Pre-order: outer chain, then
    # ``inv_zero`` (``i0``) as the sibling-Zero, then the K chain
    # (``k0, k1, ...``), terminating in ``inv_contract`` (``i1``). Names
    # are pre-assigned in pre-order so the bottom-up build below produces
    # an already-canonicalized IR.
    inv_ct = b.add_invocation("i1", primitive="contraction")
    inner: str = inv_ct
    for ax, kid in zip(reversed(k_axes), reversed(k_ids), strict=True):
        inner = b.add_iteration(kid, axis=ax, children=[inner])

    inv_zero = b.add_invocation("i0", primitive="zero")

    if outer_axes:
        outer_node = ""
        for i, (ax, nid) in enumerate(
            zip(reversed(outer_axes), reversed(outer_ids), strict=True)
        ):
            children_ids = [inv_zero, inner] if i == 0 else [outer_node]
            outer_node = b.add_iteration(nid, axis=ax, children=children_ids)
        b.set_roots([outer_node])
    else:
        # All output axes are absent (pure inner product). Zero runs once
        # before the K subtree begins; no guard needed.
        b.set_roots([inv_zero, inner])
    return b.finish(validate=True)


def _canonical_iter_ids(
    axes: Sequence[str], classification: Mapping[str, str]
) -> list[str]:
    """Assign canonical iteration-node ids to a pre-order axis sequence.

    Each axis's role (``C``/``M``/``N``/``K``) selects its prefix from
    `ROLE_PREFIX`; per-role counters keep each role's ids dense. The
    returned list is parallel to ``axes`` and contains the id that the
    iteration node over ``axes[i]`` should carry.
    """

    counters: dict[str, int] = {}
    ids: list[str] = []
    for ax in axes:
        prefix = ROLE_PREFIX[classification[ax]]
        counters.setdefault(prefix, 0)
        ids.append(f"{prefix}{counters[prefix]}")
        counters[prefix] += 1
    return ids


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


def _row_major_strides(
    axes: tuple[str, ...],
    dim_sizes: Mapping[str, int],
    element_bytes: int,
) -> dict[str, int]:
    """Compute row-major byte strides for ``axes`` in their given order.

    The last axis is the unit-stride (innermost) axis.
    """

    strides: dict[str, int] = {}
    stride = element_bytes
    for ax in reversed(axes):
        strides[ax] = stride
        stride *= dim_sizes[ax]
    return strides
