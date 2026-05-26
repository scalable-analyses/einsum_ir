"""Immutable record types for the TEIR data model."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Final

from etops.ir.dtypes import DataType

if TYPE_CHECKING:
    from etops.ir.builder import TeirBuilder

__all__ = [
    "VALID_POLICIES",
    "Axis",
    "First",
    "Guard",
    "InvocationNode",
    "IterationNode",
    "Last",
    "Primitive",
    "Schedule",
    "Teir",
    "Tensor",
    "guard",
    "is_guard",
]

#: The two iteration-policy strings accepted everywhere a policy is read.
VALID_POLICIES: Final[frozenset[str]] = frozenset({"sequential", "parallel"})


# --------------------------------------------------------------------------
# Guard
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class First:
    """Guard term that is true on the first iteration of ``axis``."""

    axis: str


@dataclass(frozen=True)
class Last:
    """Guard term that is true on the last iteration of ``axis``."""

    axis: str


#: A guard is a non-empty tuple of `First` / `Last` terms, AND-combined.
#: Build one with `guard(First("a"), Last("b"))`; use ``None`` for "no guard."
Guard = tuple["First | Last", ...]


def guard(*terms: First | Last) -> Guard:
    """Build a non-empty `Guard` from First / Last terms."""

    if not terms:
        raise ValueError(
            "guard must have at least one term; use None for an absent guard"
        )
    for term in terms:
        if not isinstance(term, First | Last):
            raise TypeError(
                f"guard term must be First or Last; got {type(term).__name__}"
            )
    return terms


def is_guard(value: object) -> bool:
    """Return True if ``value`` is a valid non-empty guard tuple."""

    return (
        isinstance(value, tuple)
        and len(value) > 0
        and all(isinstance(t, First | Last) for t in value)
    )


def format_guard(guard: Guard) -> str:
    """Render a `Guard` as the canonical textual-IR conjunction.

    Each term renders as ``first(@axis)`` or ``last(@axis)``; the terms
    are joined with `` and ``.
    """

    parts: list[str] = []
    for term in guard:
        if isinstance(term, First):
            parts.append(f"first(@{term.axis})")
        else:
            parts.append(f"last(@{term.axis})")
    return " and ".join(parts)


# --------------------------------------------------------------------------
# Tensor / Axis / Primitive
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Tensor:
    """A tensor participating in the IR.

    Attributes:
        id: Tensor identifier (``in0``, ``in1``, ``out``).
        dtype: Element data type.
    """

    id: str
    dtype: DataType


@dataclass(frozen=True)
class Axis:
    """A TEIR axis with per-tensor byte strides and offsets.

    Attributes:
        id: Axis identifier.
        extent: Positive axis extent.
        strides: ``tensor_id → byte_stride``; missing entries imply zero.
        offsets: ``tensor_id → byte_offset``; missing entries imply zero.
            Offsets are *scoped*: they enter a tile address only when the
            axis is iterated by an ancestor in the current schedule path.
            An axis with non-zero offsets cannot appear in any primitive
            role list; the validator rejects such cases.
    """

    id: str
    extent: int
    strides: Mapping[str, int] = field(default_factory=dict)
    offsets: Mapping[str, int] = field(default_factory=dict)

    def stride_on(self, tensor: str) -> int:
        """Byte stride of this axis on ``tensor``; zero if absent."""

        return int(self.strides.get(tensor, 0))

    def offset_on(self, tensor: str) -> int:
        """Byte offset of this axis on ``tensor``; zero if absent."""

        return int(self.offsets.get(tensor, 0))


@dataclass(frozen=True)
class Primitive:
    """A primitive specification.

    Attributes:
        id: Primitive identifier.
        operation: Registered operation name (``Zero``, ``Copy``, ``ReLU``,
            ``Contraction``).
        axes: Role → ordered axis ids consumed at that role.
        metadata: Operation-specific metadata (``data_type``, …).
    """

    id: str
    operation: str
    axes: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def role(self, role: str) -> tuple[str, ...]:
        """Return the ordered axis list for ``role``; empty if absent."""

        return self.axes.get(role, ())


# --------------------------------------------------------------------------
# Schedule
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class IterationNode:
    """A schedule iteration node.

    Attributes:
        id: Unique node identifier.
        axis: Axis iterated by this node.
        policy: ``"sequential"`` or ``"parallel"``.
        children: Ordered ids of child schedule nodes; must be non-empty.
        guard: Optional entry guard.
        metadata: Advisory metadata (e.g. ``threading.num_threads``).
    """

    id: str
    axis: str
    policy: str
    children: tuple[str, ...]
    guard: Guard | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class InvocationNode:
    """A schedule invocation node (leaf).

    Attributes:
        id: Unique node identifier.
        primitive: Identifier of the invoked primitive.
        guard: Optional entry guard.
        metadata: Advisory metadata.
    """

    id: str
    primitive: str
    guard: Guard | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Schedule:
    """Forest of schedule nodes.

    Attributes:
        roots: Ordered ids of root nodes.
        iterations: Map of iteration node id → `IterationNode`.
        invocations: Map of invocation node id → `InvocationNode`.
    """

    roots: tuple[str, ...] = ()
    iterations: Mapping[str, IterationNode] = field(default_factory=dict)
    invocations: Mapping[str, InvocationNode] = field(default_factory=dict)


# --------------------------------------------------------------------------
# Teir container
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Teir:
    """A full, immutable TEIR configuration."""

    tensors: Mapping[str, Tensor] = field(default_factory=dict)
    axes: Mapping[str, Axis] = field(default_factory=dict)
    primitives: Mapping[str, Primitive] = field(default_factory=dict)
    schedule: Schedule = field(default_factory=Schedule)
    name: str = ""

    @property
    def tensor_ids(self) -> tuple[str, ...]:
        """Ordered tensor identifiers."""

        return tuple(self.tensors.keys())

    @property
    def axis_ids(self) -> tuple[str, ...]:
        """Ordered axis identifiers."""

        return tuple(self.axes.keys())

    def builder(self) -> TeirBuilder:
        """Return a builder seeded with this `Teir`'s state."""

        from etops.ir.builder import TeirBuilder

        return TeirBuilder.from_teir(self)
