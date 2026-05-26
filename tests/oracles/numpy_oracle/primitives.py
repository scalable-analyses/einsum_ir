"""Primitive lowerings for the NumPy reference backend.

Each primitive operates on tile views derived from ``TileViewBuilder``.
"""

from __future__ import annotations

import numpy as np

from etops.diag import TeirLoweringError
from etops.ir import Primitive, Teir
from tests.oracles.numpy_oracle._tile import TileViewBuilder

__all__ = [
    "execute_contraction",
    "execute_copy",
    "execute_relu",
    "execute_zero",
]


def execute_zero(
    teir: Teir,
    primitive: Primitive,
    builder: TileViewBuilder,
    bases: dict[str, int],
) -> None:
    """Set the output tile to zero.

    The ``Zero`` operation expects exactly one output. By convention the
    output tensor is the **last** tensor of the IR.
    """

    out_tensor = teir.tensor_ids[-1]
    role_axes = primitive.role("M") + primitive.role("N")
    out_view = builder.tile_view(out_tensor, bases[out_tensor], role_axes)
    out_view[...] = 0


def execute_copy(
    teir: Teir,
    primitive: Primitive,
    builder: TileViewBuilder,
    bases: dict[str, int],
) -> None:
    """Copy input tile to output tile (element-wise).

    Convention: the first tensor is the input; the last is the output.
    """

    if len(teir.tensor_ids) != 2:
        msg = f"Copy expects exactly 2 tensors; got {len(teir.tensor_ids)}"
        raise TeirLoweringError(msg)
    in_tensor = teir.tensor_ids[0]
    out_tensor = teir.tensor_ids[-1]
    role_axes = primitive.role("M") + primitive.role("N")
    in_view = builder.tile_view(in_tensor, bases[in_tensor], role_axes)
    out_view = builder.tile_view(out_tensor, bases[out_tensor], role_axes)
    out_view[...] = in_view


def execute_relu(
    teir: Teir,
    primitive: Primitive,
    builder: TileViewBuilder,
    bases: dict[str, int],
) -> None:
    """Apply ``max(0, x)`` element-wise from input to output."""

    if len(teir.tensor_ids) != 2:
        msg = f"ReLU expects exactly 2 tensors; got {len(teir.tensor_ids)}"
        raise TeirLoweringError(msg)
    in_tensor = teir.tensor_ids[0]
    out_tensor = teir.tensor_ids[-1]
    role_axes = primitive.role("M") + primitive.role("N")
    in_view = builder.tile_view(in_tensor, bases[in_tensor], role_axes)
    out_view = builder.tile_view(out_tensor, bases[out_tensor], role_axes)
    out_view[...] = np.maximum(in_view, 0)


def execute_contraction(
    teir: Teir,
    primitive: Primitive,
    builder: TileViewBuilder,
    bases: dict[str, int],
) -> None:
    """Accumulate ``out += sum_K(in0 * in1)`` over M, N, K role axes.

    Conventions:
    - ``in0`` consumes M and K axes (zero stride on N).
    - ``in1`` consumes K and N axes (zero stride on M).
    - ``out`` consumes M and N axes (zero stride on K).
    """

    if len(teir.tensor_ids) != 3:
        msg = f"Contraction expects exactly 3 tensors; got {len(teir.tensor_ids)}"
        raise TeirLoweringError(msg)
    in0_tensor, in1_tensor, out_tensor = (
        teir.tensor_ids[0],
        teir.tensor_ids[1],
        teir.tensor_ids[-1],
    )
    m_axes = primitive.role("M")
    n_axes = primitive.role("N")
    k_axes = primitive.role("K")

    in0_axes = m_axes + k_axes
    in1_axes = k_axes + n_axes
    out_axes = m_axes + n_axes

    in0_tile = builder.tile_view(in0_tensor, bases[in0_tensor], in0_axes)
    in1_tile = builder.tile_view(in1_tensor, bases[in1_tensor], in1_axes)
    out_view = builder.tile_view(out_tensor, bases[out_tensor], out_axes)

    if not (m_axes or n_axes or k_axes):
        # Scalar FMA: out += in0 * in1.
        out_view += in0_tile * in1_tile
        return

    # `np.einsum` accepts an interleaved (sublist) calling convention where
    # each operand is followed by a list of integer axis labels. We assign
    # one label per unique role axis on the fly; no string subscripts and
    # no 52-letter cap.
    labels: dict[str, int] = {}

    def labels_for(axes: tuple[str, ...]) -> list[int]:
        return [labels.setdefault(a, len(labels)) for a in axes]

    contribution = np.einsum(
        in0_tile,
        labels_for(in0_axes),
        in1_tile,
        labels_for(in1_axes),
        labels_for(out_axes),
        optimize=False,
    )
    out_view += contribution
