"""Tiled Execution IR data model and builder.

The IR is implemented as a family of immutable records — tensors, axes,
primitives, and a schedule of iteration and invocation nodes — wired together
by string identifiers. Construction goes through the mutable `TeirBuilder`.
"""

from __future__ import annotations

from etops.ir._records import (
    Axis,
    First,
    Guard,
    InvocationNode,
    IterationNode,
    Last,
    Primitive,
    Schedule,
    Teir,
    Tensor,
    guard,
    is_guard,
)
from etops.ir.builder import TeirBuilder
from etops.ir.cursors import parent_map, post_order, pre_order
from etops.ir.dtypes import DataType, all_dtypes, get_dtype
from etops.ir.primitives import Operation, get_operation
from etops.ir.validate import validate

__all__ = [
    "Axis",
    "DataType",
    "First",
    "Guard",
    "InvocationNode",
    "IterationNode",
    "Last",
    "Operation",
    "Primitive",
    "Schedule",
    "Teir",
    "TeirBuilder",
    "Tensor",
    "all_dtypes",
    "get_dtype",
    "get_operation",
    "guard",
    "is_guard",
    "parent_map",
    "post_order",
    "pre_order",
    "validate",
]
