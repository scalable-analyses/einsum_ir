"""
General-purpose transformations on ``TensorOperationConfig``.

These transforms manipulate dimensions (split, fuse, reorder) while
preserving semantic correctness of the configuration. They are the
building blocks for the tileir optimizer pipeline but are independent
of any backend-specific logic.
"""

from __future__ import annotations

__all__ = [
    "split_dim",
    "fuse_dims",
    "reorder_dims",
]

from dataclasses import replace
from typing import Sequence


# ---------------------------------------------------------------------------
# Config transforms
# ---------------------------------------------------------------------------


def split_dim(
    config: object,
    dim_index: int,
    inner_size: int,
) -> object:
    """Split a dimension into an outer and inner pair.

    The dimension at *dim_index* is replaced by two dimensions:

    * **outer** at ``dim_index``     — size ``original_size // inner_size``
    * **inner** at ``dim_index + 1`` — size ``inner_size``

    Both halves inherit the original ``dim_type`` and ``exec_type``.

    Stride rule (per tensor):
        ``outer_stride = inner_size * original_stride``
        ``inner_stride = original_stride``

    A stride of 0 stays 0 in both halves (the dimension is absent from
    that tensor).

    Args:
        config: Input configuration (``TensorOperationConfig``).
        dim_index: Index of the dimension to split (0-based).
        inner_size: Size of the inner (rightmost) half.  Must be a
            positive integer that divides the original size evenly.

    Returns:
        New ``TensorOperationConfig`` with ``num_dims + 1`` dimensions.

    Raises:
        ValueError: If *inner_size* is not a positive integer, does not
            divide the dimension size, or *dim_index* is out of range.
    """
    if inner_size < 1:
        raise ValueError(f"inner_size must be >= 1, got {inner_size}.")

    num_dims = len(config.dim_sizes)
    if not 0 <= dim_index < num_dims:
        raise ValueError(
            f"dim_index {dim_index} out of range for {num_dims} dimensions."
        )

    original_size = config.dim_sizes[dim_index]
    if original_size % inner_size != 0:
        raise ValueError(
            f"inner_size {inner_size} does not divide dim_sizes[{dim_index}]"
            f" = {original_size}."
        )
    outer_size = original_size // inner_size

    # --- dim_types / exec_types / dim_sizes ---
    dim_types = list(config.dim_types)
    exec_types = list(config.exec_types)
    dim_sizes = list(config.dim_sizes)

    orig_dt = dim_types[dim_index]
    orig_et = exec_types[dim_index]

    dim_types.insert(dim_index + 1, orig_dt)
    exec_types.insert(dim_index + 1, orig_et)

    dim_sizes[dim_index] = outer_size
    dim_sizes.insert(dim_index + 1, inner_size)

    # --- strides ---
    new_levels = []
    for level in config.strides:
        new_tensors = []
        for tensor_strides in level:
            strides = list(tensor_strides)
            orig_stride = strides[dim_index]
            if orig_stride == 0:
                outer_stride = 0
                inner_stride = 0
            else:
                outer_stride = inner_size * orig_stride
                inner_stride = orig_stride
            strides[dim_index] = outer_stride
            strides.insert(dim_index + 1, inner_stride)
            new_tensors.append(tuple(strides))
        new_levels.append(tuple(new_tensors))

    return replace(
        config,
        dim_types=tuple(dim_types),
        exec_types=tuple(exec_types),
        dim_sizes=tuple(dim_sizes),
        strides=tuple(new_levels),
    )


def fuse_dims(
    config: object,
    dim_a: int,
    dim_b: int,
) -> object:
    """Fuse two dimensions into one.

    The two dimensions may be non-adjacent; intermediate dimensions are
    left in place and shift down by one index.  The fused result
    replaces ``dim_a``; ``dim_b`` is removed.

    ``dim_a`` must be less than ``dim_b``.  Both dimensions must share
    the same ``dim_type`` and ``exec_type``.  For every tensor, the
    strides must be compatible:

    * ``stride[a] == size[b] * stride[b]``, **or**
    * both strides are 0 (the dimension is absent from that tensor).

    The result is a single dimension at position ``dim_a`` with:

    * ``size = size[a] * size[b]``
    * ``stride = stride[b]`` (per tensor)

    Args:
        config: Input configuration (``TensorOperationConfig``).
        dim_a: Index of the outer (larger-stride) dimension.
        dim_b: Index of the inner (smaller-stride) dimension.
            Must satisfy ``dim_b > dim_a``.

    Returns:
        New ``TensorOperationConfig`` with ``num_dims - 1`` dimensions.

    Raises:
        ValueError: If *dim_a* or *dim_b* are out of range, *dim_b* is
            not greater than *dim_a*, dim/exec types differ, or strides
            are incompatible.
    """
    num_dims = len(config.dim_sizes)
    if not (0 <= dim_a < num_dims and 0 <= dim_b < num_dims):
        raise ValueError(
            f"dim_a={dim_a}, dim_b={dim_b} out of range for {num_dims} dimensions."
        )
    if dim_b <= dim_a:
        raise ValueError(f"dim_b ({dim_b}) must be greater than dim_a ({dim_a}).")

    if config.dim_types[dim_a] != config.dim_types[dim_b]:
        raise ValueError(
            f"Cannot fuse dims with different dim_types:"
            f" {config.dim_types[dim_a]} vs {config.dim_types[dim_b]}."
        )
    if config.exec_types[dim_a] != config.exec_types[dim_b]:
        raise ValueError(
            f"Cannot fuse dims with different exec_types:"
            f" {config.exec_types[dim_a]} vs {config.exec_types[dim_b]}."
        )

    # Validate stride compatibility for all tensors at all levels.
    size_b = config.dim_sizes[dim_b]
    for level_idx, level in enumerate(config.strides):
        for tensor_idx, tensor_strides in enumerate(level):
            stride_a = tensor_strides[dim_a]
            stride_b = tensor_strides[dim_b]
            compatible = (
                stride_a == 0 and stride_b == 0
            ) or stride_a == size_b * stride_b
            if not compatible:
                raise ValueError(
                    f"Incompatible strides for fuse at level {level_idx},"
                    f" tensor {tensor_idx}: stride[{dim_a}]={stride_a}"
                    f" != size[{dim_b}]={size_b} * stride[{dim_b}]"
                    f"={stride_b}."
                )

    # Build fused arrays.
    fused_size = config.dim_sizes[dim_a] * config.dim_sizes[dim_b]

    dim_types = list(config.dim_types)
    exec_types = list(config.exec_types)
    dim_sizes = list(config.dim_sizes)

    dim_sizes[dim_a] = fused_size
    del dim_types[dim_b]
    del exec_types[dim_b]
    del dim_sizes[dim_b]

    new_levels = []
    for level in config.strides:
        new_tensors = []
        for tensor_strides in level:
            strides = list(tensor_strides)
            # Fused stride is the inner stride (stride[b]).
            strides[dim_a] = strides[dim_b]
            del strides[dim_b]
            new_tensors.append(tuple(strides))
        new_levels.append(tuple(new_tensors))

    return replace(
        config,
        dim_types=tuple(dim_types),
        exec_types=tuple(exec_types),
        dim_sizes=tuple(dim_sizes),
        strides=tuple(new_levels),
    )


def reorder_dims(
    config: object,
    permutation: Sequence[int],
) -> object:
    """Reorder dimensions according to *permutation*.

    ``permutation`` is a sequence of length ``num_dims`` that maps new
    position ``j`` to old position ``permutation[j]``.  All parallel
    arrays (dim_types, exec_types, dim_sizes, strides) are permuted.

    Args:
        config: Input configuration (``TensorOperationConfig``).
        permutation: Old indices in the desired new order.

    Returns:
        New ``TensorOperationConfig`` with reordered dimensions.

    Raises:
        ValueError: If *permutation* length does not match the number
            of dimensions, contains out-of-range indices, or has
            duplicates.
    """
    num_dims = len(config.dim_sizes)
    if len(permutation) != num_dims:
        raise ValueError(
            f"permutation length ({len(permutation)}) must match num_dims ({num_dims})."
        )
    if set(permutation) != set(range(num_dims)):
        raise ValueError(
            f"permutation must contain each index in"
            f" range({num_dims}) exactly once, got {tuple(permutation)}."
        )

    dim_types = tuple(config.dim_types[i] for i in permutation)
    exec_types = tuple(config.exec_types[i] for i in permutation)
    dim_sizes = tuple(config.dim_sizes[i] for i in permutation)

    new_levels = []
    for level in config.strides:
        new_tensors = []
        for tensor_strides in level:
            new_tensors.append(tuple(tensor_strides[i] for i in permutation))
        new_levels.append(tuple(new_tensors))

    return replace(
        config,
        dim_types=dim_types,
        exec_types=exec_types,
        dim_sizes=dim_sizes,
        strides=tuple(new_levels),
    )
