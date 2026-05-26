"""Primitive operations.

A closed catalog of the four TEIR operations: ``Zero``, ``Copy``,
``ReLU``, ``Contraction``. Each declares the set of roles a primitive of
this operation may carry and the metadata schema it accepts. Backends
register lowerings separately (see ``etops.lowering``); operations stay
backend-neutral.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

__all__ = ["Operation", "get_operation"]


@dataclass(frozen=True)
class Operation:
    """Static specification of a primitive operation.

    Attributes:
        name: Canonical name.
        roles: Ordered tuple of role names this operation accepts. Every
            primitive of this operation must list every role (with an
            axis list of any length, including empty).
        metadata_schema: Names of metadata keys the operation accepts,
            with the required type for each. Keys not listed are rejected.
    """

    name: str
    roles: tuple[str, ...] = ()
    metadata_schema: Mapping[str, type] = field(default_factory=dict)


_UNARY_ROLES: tuple[str, ...] = ("M", "N")
_CONTRACTION_ROLES: tuple[str, ...] = ("M", "N", "K")
_DTYPE_META: dict[str, type] = {"data_type": str}


_CATALOG: tuple[Operation, ...] = (
    Operation(name="Zero", roles=_UNARY_ROLES, metadata_schema=_DTYPE_META),
    Operation(name="Copy", roles=_UNARY_ROLES, metadata_schema=_DTYPE_META),
    Operation(name="ReLU", roles=_UNARY_ROLES, metadata_schema=_DTYPE_META),
    Operation(
        name="Contraction",
        roles=_CONTRACTION_ROLES,
        metadata_schema=_DTYPE_META,
    ),
)


_OPERATIONS: dict[str, Operation] = {op.name: op for op in _CATALOG}


def get_operation(name: str) -> Operation:
    """Look up a registered `Operation` by name.

    Raises:
        KeyError: If no `Operation` is registered for ``name``.
    """

    return _OPERATIONS[name]
