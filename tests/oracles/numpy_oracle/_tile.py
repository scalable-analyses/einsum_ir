"""Tile-view construction for the NumPy reference backend.

Tiles are exposed as writable strided NumPy views built via
`np.lib.stride_tricks.as_strided`. The view's shape and byte strides come
directly from the IR's per-axis stride table, so reads and writes touch
the same buffer the schedule is iterating. A one-shot upfront bounds
check at view construction prevents the out-of-bounds aliasing that
`as_strided` would otherwise permit silently.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
from numpy.lib.stride_tricks import as_strided

from etops.ir import Teir

__all__ = ["TileViewBuilder"]


class TileViewBuilder:
    """Reusable tile-view helper for a single `Teir` configuration."""

    def __init__(self, teir: Teir, flats: Mapping[str, np.ndarray]) -> None:
        self._teir = teir
        self._flats = dict(flats)

    def base_byte_address(
        self,
        tensor: str,
        ancestor_indices: Mapping[str, int],
    ) -> int:
        """Byte offset of the current tile within ``tensor``.

        Sum over ancestor iteration axes of ``offset + stride * index``
        for ``tensor``. Axes absent from ``tensor`` contribute zero.
        """

        total = 0
        for axis_id, idx in ancestor_indices.items():
            axis = self._teir.axes[axis_id]
            total += axis.offsets.get(tensor, 0)
            total += axis.strides.get(tensor, 0) * idx
        return total

    def tile_view(
        self,
        tensor: str,
        base_byte_offset: int,
        primitive_axes: tuple[str, ...],
    ) -> np.ndarray:
        """Return a writable strided view of the current tile.

        Shape is the extents of ``primitive_axes`` in order; byte strides
        come from the tensor's axis stride table. An empty
        ``primitive_axes`` yields a 0-D view (a scalar tile).

        Raises:
            ValueError: When the base byte offset is not element-aligned,
                an axis byte stride is not element-aligned, or the tile
                would extend past the buffer.
        """

        flat = self._flats[tensor]
        itemsize = self._teir.tensors[tensor].dtype.bytes
        if base_byte_offset % itemsize != 0:
            msg = (
                f"tile base byte offset {base_byte_offset} is not a multiple"
                f" of element width {itemsize} for tensor {tensor!r}"
            )
            raise ValueError(msg)
        base_elem = base_byte_offset // itemsize

        shape: list[int] = []
        byte_strides: list[int] = []
        max_elem = base_elem
        for axis_id in primitive_axes:
            axis = self._teir.axes[axis_id]
            stride_bytes = axis.strides.get(tensor, 0)
            if stride_bytes % itemsize != 0:
                msg = (
                    f"axis {axis_id!r} byte stride {stride_bytes} on tensor"
                    f" {tensor!r} is not a multiple of element width {itemsize}"
                )
                raise ValueError(msg)
            shape.append(axis.extent)
            byte_strides.append(stride_bytes)
            if stride_bytes > 0 and axis.extent > 0:
                max_elem += (stride_bytes // itemsize) * (axis.extent - 1)

        if max_elem >= len(flat):
            msg = (
                f"tile view for tensor {tensor!r} extends past buffer:"
                f" max element index {max_elem}, buffer length {len(flat)}"
            )
            raise ValueError(msg)

        return as_strided(
            flat[base_elem:],
            shape=tuple(shape),
            strides=tuple(byte_strides),
            writeable=True,
        )
