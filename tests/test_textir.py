"""Tests for the textual IR (round-trip, error reporting, edge cases)."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pytest

import etops
from etops.emit import einsum
from etops.ir import Teir, TeirBuilder
from etops.textir import dump, parse


def _teir_with_metadata(metadata: Mapping[str, Any]) -> Teir:
    """Build a tiny IR whose iteration node carries the requested metadata.

    Iteration / invocation node metadata is the unconstrained channel for
    advisory backend hints; this fixture uses it to exercise textual-IR
    metadata round-tripping without depending on a per-primitive schema.
    """

    b = TeirBuilder().set_name("meta_round_trip")
    b.add_tensor("out", dtype="f32")
    b.add_axis("a", extent=4, strides_by_tensor={"out": 4})
    b.add_primitive(
        "zero", operation="Zero", axes={"M": [], "N": []}, metadata={"data_type": "f32"}
    )
    inv = b.add_invocation("inv_zero", primitive="zero")
    root = b.add_iteration("iter_a", axis="a", children=[inv], metadata=metadata)
    b.set_roots([root])
    return b.finish(validate=True)


class TestRoundTrip:
    """Header and optional-header behavior. (Property tests in
    `tests/test_property.py::TestRoundTripProperty` sweep arbitrary IRs.)"""

    def test_dump_includes_format_header(self) -> None:
        """The serialized form starts with ``teir-format 1.0``."""
        t = einsum("ab->ba", dim_sizes=dict(a=2, b=3))
        assert dump(t).splitlines()[0] == "teir-format 1.0"

    def test_parse_without_header(self) -> None:
        """The header is optional."""
        t = einsum("ab->ba", dim_sizes=dict(a=2, b=3))
        text = dump(t).split("\n", 2)[2]  # drop the header line
        assert parse(text) == t


class TestMetadataRoundTrip:
    """Metadata values round-trip through the textual IR."""

    def test_bool_metadata(self) -> None:
        """Booleans serialize as ``true`` / ``false`` and round-trip."""

        t = _teir_with_metadata({"flag": True, "other": False})
        round_tripped = parse(dump(t))
        meta = round_tripped.schedule.iterations["iter_a"].metadata
        assert meta["flag"] is True
        assert meta["other"] is False

    def test_int_metadata(self) -> None:
        """Integer metadata round-trips as int."""

        t = _teir_with_metadata({"tile": 32})
        meta = parse(dump(t)).schedule.iterations["iter_a"].metadata
        assert meta["tile"] == 32
        assert isinstance(meta["tile"], int)

    def test_string_with_space_is_quoted(self) -> None:
        """A string containing a space serializes with quotes."""

        t = _teir_with_metadata({"label": "with space"})
        text = dump(t)
        assert '"with space"' in text
        meta = parse(text).schedule.iterations["iter_a"].metadata
        assert meta["label"] == "with space"

    def test_escape_sequences_round_trip(self) -> None:
        """Backslash, quote, newline, and tab round-trip via escapes."""

        value = 'a\\b"c\nd\te'
        t = _teir_with_metadata({"path": value})
        text = dump(t)
        # Backslash escapes appear in the serialized form.
        assert '"a\\\\b\\"c\\nd\\te"' in text
        meta = parse(text).schedule.iterations["iter_a"].metadata
        assert meta["path"] == value

    def test_true_string_is_quoted(self) -> None:
        """A metadata string equal to ``true`` must be quoted so the parser
        does not treat it as a bool."""

        t = _teir_with_metadata({"label": "true"})
        text = dump(t)
        assert '"true"' in text
        meta = parse(text).schedule.iterations["iter_a"].metadata
        assert meta["label"] == "true"


class TestNodeMetadataRoundTrip:
    """Iteration / invocation node metadata round-trips through the textual IR."""

    def _teir_with_node_metadata(
        self,
        iter_metadata: Mapping[str, Any] | None = None,
        inv_metadata: Mapping[str, Any] | None = None,
    ) -> Teir:
        b = TeirBuilder().set_name("node_meta")
        b.add_tensor("out", dtype="f32")
        b.add_axis("a", extent=4, strides_by_tensor={"out": 4})
        b.add_primitive(
            "zero",
            operation="Zero",
            axes={"M": [], "N": []},
            metadata={"data_type": "f32"},
        )
        inv = b.add_invocation(
            "inv_zero", primitive="zero", metadata=inv_metadata or {}
        )
        root = b.add_iteration(
            "iter_a", axis="a", children=[inv], metadata=iter_metadata or {}
        )
        b.set_roots([root])
        return b.finish(validate=True)

    def test_iteration_node_metadata(self) -> None:
        """An iteration-node metadata map round-trips through textir."""

        t = self._teir_with_node_metadata(iter_metadata={"threading.num_threads": 4})
        round_tripped = parse(dump(t))
        meta = round_tripped.schedule.iterations["iter_a"].metadata
        assert meta["threading.num_threads"] == 4

    def test_invocation_node_metadata(self) -> None:
        """An invocation-node metadata map round-trips through textir."""

        t = self._teir_with_node_metadata(inv_metadata={"trace": True, "note": "init"})
        round_tripped = parse(dump(t))
        meta = round_tripped.schedule.invocations["inv_zero"].metadata
        assert meta["trace"] is True
        assert meta["note"] == "init"

    def test_full_teir_equality_with_node_metadata(self) -> None:
        """`parse(dump(t)) == t` holds when iter and invoke carry metadata."""

        t = self._teir_with_node_metadata(
            iter_metadata={"threading.num_threads": 2},
            inv_metadata={"note": "leaf"},
        )
        assert parse(dump(t)) == t


class TestIdShape:
    """Builder rejects identifier shapes the textual lexer cannot encode."""

    def test_digit_prefixed_tensor_id_rejected(self) -> None:
        """A tensor id starting with a digit is rejected by the builder."""

        b = TeirBuilder()
        with pytest.raises(etops.TeirEmissionError, match="must match"):
            b.add_tensor("0in", dtype="f32")

    def test_digit_prefixed_axis_id_rejected(self) -> None:
        """An axis id starting with a digit is rejected by the builder."""

        b = TeirBuilder()
        b.add_tensor("in0", dtype="f32")
        with pytest.raises(etops.TeirEmissionError, match="must match"):
            b.add_axis("0a", extent=4)

    def test_invalid_character_in_id_rejected(self) -> None:
        """An axis id containing a forbidden character is rejected."""

        b = TeirBuilder()
        b.add_tensor("in0", dtype="f32")
        with pytest.raises(etops.TeirEmissionError, match="must match"):
            b.add_axis("a/b", extent=4)


class TestParserDiagnostics:
    """The parser raises informative errors."""

    def test_unknown_top_level_keyword(self) -> None:
        """An unknown keyword inside a teir block is rejected."""
        text = """
teir @bad {
  garbage
}
"""
        with pytest.raises(etops.TeirEmissionError, match="unknown top-level"):
            parse(text)

    def test_unsupported_format_version(self) -> None:
        """Unsupported format versions are rejected."""
        text = """
teir-format 99.0

teir @bad {
}
"""
        with pytest.raises(etops.TeirEmissionError, match="unsupported textir format"):
            parse(text)

    def test_unclosed_block(self) -> None:
        """Unclosed blocks are rejected."""
        text = """
teir @bad {
  tensor %in0 : f32
"""
        with pytest.raises(etops.TeirEmissionError, match="unexpected end of input"):
            parse(text)
