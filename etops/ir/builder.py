"""Mutable builder that produces immutable `Teir` instances."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import replace
from typing import Any

from etops.diag import TeirEmissionError
from etops.ir._records import (
    VALID_POLICIES,
    Axis,
    Guard,
    InvocationNode,
    IterationNode,
    Primitive,
    Schedule,
    Teir,
    Tensor,
    is_guard,
)
from etops.ir.dtypes import DataType, get_dtype
from etops.ir.primitives import Operation, get_operation

__all__ = ["TeirBuilder"]

_ID_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_.\-]*")


class TeirBuilder:
    """Mutable accumulator for an immutable `Teir`.

    A builder is the only legitimate way to construct or mutate IR state.
    Methods enforce cheap invariants (id uniqueness, type checks, required
    keys, schema arity) at insertion time. Global invariants (axis
    reachability of guards, schedule acyclicity) are enforced by
    `etops.ir.validate.validate`.

    All mutators raise `TeirEmissionError` on invalid input.
    """

    def __init__(self) -> None:
        self._tensors: dict[str, Tensor] = {}
        self._axes: dict[str, Axis] = {}
        self._primitives: dict[str, Primitive] = {}
        self._iterations: dict[str, IterationNode] = {}
        self._invocations: dict[str, InvocationNode] = {}
        self._roots: list[str] = []
        self._name: str = ""

    # ----------------------------------------------------------------------
    # construction helpers
    # ----------------------------------------------------------------------

    @classmethod
    def from_teir(cls, teir: Teir) -> TeirBuilder:
        """Seed a builder from an existing `Teir`."""

        b = cls()
        b._name = teir.name
        b._tensors = dict(teir.tensors)
        b._axes = dict(teir.axes)
        b._primitives = dict(teir.primitives)
        b._iterations = dict(teir.schedule.iterations)
        b._invocations = dict(teir.schedule.invocations)
        b._roots = list(teir.schedule.roots)
        return b

    @property
    def name(self) -> str:
        """Symbolic name carried by the produced `Teir`."""

        return self._name

    def set_name(self, name: str) -> TeirBuilder:
        """Set the symbolic name. Returns ``self`` to support chaining."""

        if not isinstance(name, str):
            raise TypeError(f"name must be a string; got {type(name).__name__}")
        self._name = name
        return self

    # ----------------------------------------------------------------------
    # tensors
    # ----------------------------------------------------------------------

    def add_tensor(self, id: str, *, dtype: str | DataType) -> str:
        """Add a tensor. Returns its id."""

        _check_id(id)
        if id in self._tensors:
            raise TeirEmissionError(f"tensor id already in use: {id!r}")
        if isinstance(dtype, str):
            dtype_obj = get_dtype(dtype)
        elif isinstance(dtype, DataType):
            try:
                catalog_entry = get_dtype(dtype.name)
            except KeyError as exc:
                raise TeirEmissionError(
                    f"dtype {dtype!r} is not in the closed catalog"
                ) from exc
            if catalog_entry != dtype:
                raise TeirEmissionError(
                    f"dtype {dtype!r} does not match the closed-catalog entry"
                    f" {catalog_entry!r}"
                )
            dtype_obj = dtype
        else:
            raise TypeError(
                f"dtype must be a string or DataType; got {type(dtype).__name__}"
            )
        self._tensors[id] = Tensor(id=id, dtype=dtype_obj)
        return id

    # ----------------------------------------------------------------------
    # axes
    # ----------------------------------------------------------------------

    def add_axis(
        self,
        id: str,
        *,
        extent: int,
        strides_by_tensor: Mapping[str, int] | None = None,
        offsets_by_tensor: Mapping[str, int] | None = None,
    ) -> str:
        """Add an axis. Strides and offsets are in **bytes**."""

        if not isinstance(extent, int) or isinstance(extent, bool):
            raise TypeError("extent must be an int")
        if extent <= 0:
            raise ValueError(f"axis extent must be positive; got {extent}")

        _check_id(id)
        if id in self._axes:
            raise TeirEmissionError(f"axis id already in use: {id!r}")
        strides = self._coerce_strides(id, strides_by_tensor)
        offsets = self._coerce_offsets(id, offsets_by_tensor)
        self._axes[id] = Axis(
            id=id, extent=int(extent), strides=strides, offsets=offsets
        )
        return id

    def has_axis(self, axis_id: str) -> bool:
        """Return True if ``axis_id`` names an existing axis."""

        return axis_id in self._axes

    def remove_axis(self, axis_id: str) -> None:
        """Drop an axis from the builder.

        Dangling references in iteration nodes, primitive role lists, or
        guards are caught by ``finish(validate=True)``.
        """

        self._require_axis(axis_id)
        self._axes.pop(axis_id, None)

    # ----------------------------------------------------------------------
    # primitives
    # ----------------------------------------------------------------------

    def add_primitive(
        self,
        id: str,
        *,
        operation: str,
        axes: Mapping[str, Sequence[str]] | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> str:
        """Add a primitive. Returns its id."""

        try:
            op = get_operation(operation)
        except KeyError as exc:
            raise TeirEmissionError(
                f"unknown primitive operation {operation!r}"
            ) from exc

        _check_id(id)
        if id in self._primitives:
            raise TeirEmissionError(f"primitive id already in use: {id!r}")

        axes_in = dict(axes) if axes else {}
        canonical: dict[str, tuple[str, ...]] = {}
        for role in op.roles:
            role_axes = axes_in.pop(role, None)
            if role_axes is None:
                raise TeirEmissionError(
                    f"primitive {id!r} ({operation}) missing role {role!r}"
                )
            role_axes_list = tuple(str(a) for a in role_axes)
            for axid in role_axes_list:
                if axid not in self._axes:
                    raise TeirEmissionError(
                        f"primitive {id!r} role {role!r} references unknown axis {axid!r}"
                    )
            canonical[role] = role_axes_list

        if axes_in:
            extras = sorted(axes_in.keys())
            raise TeirEmissionError(
                f"primitive {id!r} ({operation}) declares unexpected roles: {extras}"
            )

        meta_canonical: dict[str, Any] = {}
        for key, value in (metadata or {}).items():
            self._check_metadata_value(id, op, key, value)
            meta_canonical[key] = value

        self._primitives[id] = Primitive(
            id=id, operation=op.name, axes=canonical, metadata=meta_canonical
        )
        return id

    def primitive_role_axes(self, primitive_id: str, role: str) -> tuple[str, ...]:
        """Return the axis list assigned to ``role`` of ``primitive_id``."""

        entry = self._require_primitive(primitive_id)
        op = get_operation(entry.operation)
        if role not in op.roles and role not in entry.axes:
            raise TeirEmissionError(
                f"primitive {entry.id!r} ({entry.operation}) has no role {role!r}"
            )
        return tuple(entry.axes.get(role, ()))

    def set_primitive_role_axes(
        self,
        primitive_id: str,
        role: str,
        axes: Sequence[str],
    ) -> None:
        """Replace the axis list assigned to ``role`` of ``primitive_id``."""

        entry = self._require_primitive(primitive_id)
        op = get_operation(entry.operation)
        if role not in op.roles:
            raise TeirEmissionError(
                f"primitive {entry.id!r} ({entry.operation}) does not accept role {role!r}"
            )
        new_axes = tuple(str(a) for a in axes)
        for axid in new_axes:
            if axid not in self._axes:
                raise TeirEmissionError(
                    f"primitive {entry.id!r} role {role!r}"
                    f" references unknown axis {axid!r}"
                )
        new_role_map = dict(entry.axes)
        new_role_map[role] = new_axes
        self._primitives[entry.id] = replace(entry, axes=new_role_map)

    # ----------------------------------------------------------------------
    # schedule nodes
    # ----------------------------------------------------------------------

    def add_iteration(
        self,
        id: str,
        *,
        axis: str,
        policy: str = "sequential",
        children: Sequence[str],
        guard: Guard | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> str:
        """Add an iteration node. Returns its id."""

        if policy not in VALID_POLICIES:
            raise TeirEmissionError(
                f"iteration policy must be one of {sorted(VALID_POLICIES)};"
                f" got {policy!r}"
            )
        if axis not in self._axes:
            raise TeirEmissionError(f"iteration node references unknown axis {axis!r}")
        _check_id(id)
        if id in self._iterations or id in self._invocations:
            raise TeirEmissionError(f"node id already in use: {id!r}")
        kids = self._validate_children_list(id, children)
        if guard is not None and not is_guard(guard):
            raise TypeError(
                f"guard must be a non-empty tuple of First/Last terms or None;"
                f" got {type(guard).__name__}"
            )
        self._iterations[id] = IterationNode(
            id=id,
            axis=axis,
            policy=policy,
            children=tuple(kids),
            guard=guard,
            metadata=dict(metadata) if metadata else {},
        )
        return id

    def add_invocation(
        self,
        id: str,
        *,
        primitive: str,
        guard: Guard | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> str:
        """Add an invocation node. Returns its id."""

        if primitive not in self._primitives:
            raise TeirEmissionError(
                f"invocation node references unknown primitive {primitive!r}"
            )
        if guard is not None and not is_guard(guard):
            raise TypeError(
                f"guard must be a non-empty tuple of First/Last terms or None;"
                f" got {type(guard).__name__}"
            )
        _check_id(id)
        if id in self._iterations or id in self._invocations:
            raise TeirEmissionError(f"node id already in use: {id!r}")
        self._invocations[id] = InvocationNode(
            id=id,
            primitive=primitive,
            guard=guard,
            metadata=dict(metadata) if metadata else {},
        )
        return id

    def has_node(self, node_id: str) -> bool:
        """Return True if ``node_id`` names a schedule node."""

        return node_id in self._iterations or node_id in self._invocations

    def iteration_children(self, node_id: str) -> tuple[str, ...]:
        """Return the ordered children of an iteration node."""

        return self._require_iteration(node_id).children

    def remove_iteration(
        self,
        node_id: str,
        *,
        reparent_children: bool = True,
    ) -> None:
        """Drop an iteration node.

        When ``reparent_children`` is True (the default) the removed node's
        children take its place in its parent's children list (or in the
        ``roots`` list if it was a root).
        """

        entry = self._require_iteration(node_id)
        if reparent_children:
            children = list(entry.children)
        else:
            if entry.children:
                raise TeirEmissionError(
                    f"cannot remove iteration node {entry.id!r} without reparenting:"
                    f" it still has children {list(entry.children)!r}"
                )
            children = []
        del self._iterations[entry.id]
        self._splice_in_parents(entry.id, children)

    def set_iteration_children(
        self,
        node_id: str,
        children: Sequence[str],
    ) -> None:
        """Replace the children list of an iteration node."""

        entry = self._require_iteration(node_id)
        kids = self._validate_children_list(entry.id, children)
        self._iterations[entry.id] = replace(entry, children=tuple(kids))

    def set_iteration_guard(self, node_id: str, guard: Guard | None) -> None:
        """Set or clear the guard on an iteration node."""

        entry = self._require_iteration(node_id)
        if guard is not None and not is_guard(guard):
            raise TypeError(
                f"guard must be a non-empty tuple of First/Last terms or None;"
                f" got {type(guard).__name__}"
            )
        self._iterations[entry.id] = replace(entry, guard=guard)

    def set_iteration_policy(self, node_id: str, policy: str) -> None:
        """Set the iteration policy (``sequential`` / ``parallel``)."""

        entry = self._require_iteration(node_id)
        if policy not in VALID_POLICIES:
            raise TeirEmissionError(
                f"iteration policy must be one of {sorted(VALID_POLICIES)};"
                f" got {policy!r}"
            )
        self._iterations[entry.id] = replace(entry, policy=policy)

    def set_iteration_axis(self, node_id: str, axis_id: str) -> None:
        """Re-target an iteration node at a different axis."""

        entry = self._require_iteration(node_id)
        self._require_axis(axis_id)
        self._iterations[entry.id] = replace(entry, axis=axis_id)

    def iteration_axis(self, node_id: str) -> str:
        """Return the axis iterated by ``node_id``."""

        return self._require_iteration(node_id).axis

    def iteration_policy(self, node_id: str) -> str:
        """Return the iteration policy of ``node_id``."""

        return self._require_iteration(node_id).policy

    def set_iteration_metadata(
        self,
        node_id: str,
        metadata: Mapping[str, Any] | None,
    ) -> None:
        """Replace the metadata dict on an iteration node."""

        entry = self._require_iteration(node_id)
        self._iterations[entry.id] = replace(
            entry, metadata=dict(metadata) if metadata else {}
        )

    def roots(self) -> tuple[str, ...]:
        """Return the current ordered roots list."""

        return tuple(self._roots)

    def set_invocation_guard(self, node_id: str, guard: Guard | None) -> None:
        """Set or clear the guard on an invocation node."""

        entry = self._require_invocation(node_id)
        if guard is not None and not is_guard(guard):
            raise TypeError(
                f"guard must be a non-empty tuple of First/Last terms or None;"
                f" got {type(guard).__name__}"
            )
        self._invocations[entry.id] = replace(entry, guard=guard)

    def set_roots(self, roots: Iterable[str]) -> TeirBuilder:
        """Set the ordered list of schedule roots."""

        rs = [str(r) for r in roots]
        for rid in rs:
            if rid not in self._iterations and rid not in self._invocations:
                raise TeirEmissionError(
                    f"root references unknown schedule node {rid!r}"
                )
        self._roots = rs
        return self

    # ----------------------------------------------------------------------
    # finalize
    # ----------------------------------------------------------------------

    def finish(self, validate: bool = True) -> Teir:
        """Build an immutable `Teir`."""

        schedule = Schedule(
            roots=tuple(self._roots),
            iterations=dict(self._iterations),
            invocations=dict(self._invocations),
        )
        teir = Teir(
            tensors=dict(self._tensors),
            axes=dict(self._axes),
            primitives=dict(self._primitives),
            schedule=schedule,
            name=self._name,
        )
        if validate:
            from etops.ir.validate import validate as _validate

            _validate(teir)
        return teir

    # ----------------------------------------------------------------------
    # internals
    # ----------------------------------------------------------------------

    def _coerce_strides(
        self,
        axis_id: str,
        strides_by_tensor: Mapping[str, int] | None,
    ) -> dict[str, int]:
        return self._coerce_per_tensor(
            axis_id, strides_by_tensor, "stride", non_negative=True
        )

    def _coerce_offsets(
        self,
        axis_id: str,
        offsets_by_tensor: Mapping[str, int] | None,
    ) -> dict[str, int]:
        return self._coerce_per_tensor(
            axis_id, offsets_by_tensor, "offset", non_negative=False
        )

    def _coerce_per_tensor(
        self,
        axis_id: str,
        values: Mapping[str, int] | None,
        kind: str,
        *,
        non_negative: bool,
    ) -> dict[str, int]:
        out: dict[str, int] = {}
        if not values:
            return out
        for tname, value in values.items():
            if tname not in self._tensors:
                raise TeirEmissionError(
                    f"axis {axis_id!r} references unknown tensor {tname!r}"
                )
            if not isinstance(value, int) or isinstance(value, bool):
                raise TypeError(
                    f"{kind} for tensor {tname!r} must be an int;"
                    f" got {type(value).__name__}"
                )
            if non_negative and value < 0:
                raise ValueError(
                    f"{kind} for tensor {tname!r} on axis {axis_id!r}"
                    f" must be non-negative; got {value}"
                )
            if value != 0:
                out[tname] = value
        return out

    def _check_metadata_value(
        self,
        primitive_id: str,
        op: Operation,
        key: str,
        value: Any,
    ) -> None:
        expected = op.metadata_schema.get(key)
        if expected is None:
            allowed = sorted(op.metadata_schema.keys())
            raise TeirEmissionError(
                f"primitive {primitive_id!r} ({op.name}) metadata key"
                f" {key!r} is not in the operation's schema; allowed"
                f" keys: {allowed or '[]'}"
            )
        if not isinstance(value, expected):
            raise TeirEmissionError(
                f"primitive {primitive_id!r} metadata key {key!r} expected"
                f" {expected.__name__}; got {type(value).__name__}"
            )

    def _splice_in_parents(
        self,
        node_id: str,
        replacement: Sequence[str],
    ) -> None:
        for pid, entry in list(self._iterations.items()):
            if node_id not in entry.children:
                continue
            new_list: list[str] = []
            for c in entry.children:
                if c == node_id:
                    new_list.extend(replacement)
                else:
                    new_list.append(c)
            self._iterations[pid] = replace(entry, children=tuple(new_list))
        if node_id in self._roots:
            new_roots: list[str] = []
            for r in self._roots:
                if r == node_id:
                    new_roots.extend(replacement)
                else:
                    new_roots.append(r)
            self._roots = new_roots

    def _require_axis(self, axis_id: str) -> Axis:
        entry = self._axes.get(axis_id)
        if entry is None:
            raise TeirEmissionError(f"unknown axis {axis_id!r}")
        return entry

    def _require_primitive(self, primitive_id: str) -> Primitive:
        entry = self._primitives.get(primitive_id)
        if entry is None:
            raise TeirEmissionError(f"unknown primitive {primitive_id!r}")
        return entry

    def _require_iteration(self, node_id: str) -> IterationNode:
        entry = self._iterations.get(node_id)
        if entry is None:
            raise TeirEmissionError(f"unknown iteration {node_id!r}")
        return entry

    def _require_invocation(self, node_id: str) -> InvocationNode:
        entry = self._invocations.get(node_id)
        if entry is None:
            raise TeirEmissionError(f"unknown invocation {node_id!r}")
        return entry

    def _validate_children_list(
        self,
        parent_id: str,
        children: Iterable[str] | None,
    ) -> list[str]:
        kids = [str(c) for c in children] if children else []
        if not kids:
            raise TeirEmissionError(
                f"iteration node {parent_id!r} must have at least one child"
            )
        seen: set[str] = set()
        for cid in kids:
            if cid in seen:
                raise TeirEmissionError(
                    f"iteration node {parent_id!r} children contain duplicate {cid!r}"
                )
            seen.add(cid)
            if cid not in self._iterations and cid not in self._invocations:
                raise TeirEmissionError(
                    f"iteration node {parent_id!r} references unknown child {cid!r}"
                )
            if cid == parent_id:
                raise TeirEmissionError(
                    f"iteration node {parent_id!r} cannot be its own child"
                )
        return kids

    def claim_unused_axis_id(self, base: str) -> str:
        """Return an axis id not currently in use, derived from ``base``.

        Returns ``base`` if it is free; otherwise appends an ``_{counter}``
        suffix, incrementing until a free id is found.
        """

        if base not in self._axes:
            return base
        i = 1
        while f"{base}_{i}" in self._axes:
            i += 1
        return f"{base}_{i}"

    def claim_unused_node_id(self, base: str) -> str:
        """Return a schedule-node id not currently in use, derived from ``base``.

        Returns ``base`` if it is free; otherwise appends an ``_{counter}``
        suffix, incrementing until a free id is found.
        """

        if base not in self._iterations and base not in self._invocations:
            return base
        i = 1
        while f"{base}_{i}" in self._iterations or f"{base}_{i}" in self._invocations:
            i += 1
        return f"{base}_{i}"


def _check_id(requested: object) -> None:
    """Validate an identifier shape.

    The textual IR requires identifiers to start with an ASCII letter or
    underscore and to consist of ASCII letters, digits, underscore, hyphen,
    or dot. Keeping the builder in lock-step with the lexer guarantees
    ``parse(dump(t)) == t`` for every well-formed `Teir`.
    """

    if not isinstance(requested, str) or not requested:
        raise TypeError(f"id must be a non-empty string; got {requested!r}")
    if not _ID_RE.fullmatch(requested):
        raise TeirEmissionError(
            f"id {requested!r} must match [A-Za-z_][A-Za-z0-9_.\\-]*"
        )
