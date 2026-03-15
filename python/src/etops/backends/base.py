"""
Base types for backend implementations.

This module defines the protocol that all compiled operations must implement.
"""

from typing import Protocol, runtime_checkable
import numpy as np


@runtime_checkable
class CompiledOperation(Protocol):
    """
    Protocol for compiled tensor operations.

    All backends must return objects that implement this protocol.

    Example:
        >>> op = etops.compile(config)
        >>> op.execute(in0, in1, out)
    """

    def execute(
        self,
        in0: np.ndarray,
        in1: np.ndarray | None,
        out: np.ndarray
    ) -> None:
        """
        Execute the tensor operation.

        Args:
            in0: First input tensor
            in1: Second input tensor (None for unary operations)
            out: Output tensor (must be pre-allocated)
        """
        ...


__all__ = ["CompiledOperation"]
