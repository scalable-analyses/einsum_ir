"""Tests for the immutable IR and the builder."""

from __future__ import annotations

import pytest

import etops
from etops.ir import (
    First,
    Teir,
    TeirBuilder,
    guard,
    pre_order,
    validate,
)


class TestBuilder:
    """Builder enforces cheap invariants and produces immutable `Teir`."""

    def test_minimal_unary_copy(self) -> None:
        """The smallest non-trivial unary IR validates cleanly."""
        b = TeirBuilder()
        b.add_tensor("in0", dtype="f32")
        b.add_tensor("out", dtype="f32")
        b.add_axis("a", extent=8, strides_by_tensor={"in0": 4, "out": 4})
        b.add_primitive(
            "copy",
            operation="Copy",
            axes={"M": [], "N": []},
            metadata={"data_type": "f32"},
        )
        inv = b.add_invocation("inv", primitive="copy")
        itr = b.add_iteration("it", axis="a", children=[inv])
        b.set_roots([itr])
        teir = b.finish()
        assert isinstance(teir, Teir)
        assert teir.tensor_ids == ("in0", "out")
        assert teir.axis_ids == ("a",)

    def test_axis_extent_positive(self) -> None:
        """Extent must be positive."""
        b = TeirBuilder()
        b.add_tensor("in0", dtype="f32")
        with pytest.raises(ValueError, match="positive"):
            b.add_axis("a", extent=0)

    def test_axis_stride_non_negative(self) -> None:
        """Stride must be non-negative."""
        b = TeirBuilder()
        b.add_tensor("in0", dtype="f32")
        with pytest.raises(ValueError, match="non-negative"):
            b.add_axis("a", extent=4, strides_by_tensor={"in0": -1})

    def test_unknown_tensor_in_axis_stride(self) -> None:
        """Stride for an unknown tensor raises emission error."""
        b = TeirBuilder()
        with pytest.raises(etops.TeirEmissionError):
            b.add_axis("a", extent=4, strides_by_tensor={"ghost": 4})

    def test_primitive_missing_required_role(self) -> None:
        """Required roles must be declared, even if empty."""
        b = TeirBuilder()
        b.add_tensor("in0", dtype="f32")
        b.add_tensor("out", dtype="f32")
        with pytest.raises(etops.TeirEmissionError, match=r"missing role 'N'"):
            b.add_primitive(
                "broken",
                operation="Copy",
                axes={"M": []},
                metadata={"data_type": "f32"},
            )

    def test_iteration_node_requires_child(self) -> None:
        """Iteration nodes must have at least one child."""
        b = TeirBuilder()
        b.add_tensor("in0", dtype="f32")
        b.add_tensor("out", dtype="f32")
        b.add_axis("a", extent=4, strides_by_tensor={"in0": 4, "out": 4})
        with pytest.raises(etops.TeirEmissionError, match="at least one child"):
            b.add_iteration("it", axis="a", children=[])


class TestValidate:
    """Global validation reports all violations."""

    def _good_teir(self) -> Teir:
        b = TeirBuilder()
        b.add_tensor("in0", dtype="f32")
        b.add_tensor("out", dtype="f32")
        b.add_axis("a", extent=8, strides_by_tensor={"in0": 4, "out": 4})
        b.add_primitive(
            "copy",
            operation="Copy",
            axes={"M": [], "N": []},
            metadata={"data_type": "f32"},
        )
        inv = b.add_invocation("inv", primitive="copy")
        itr = b.add_iteration("it", axis="a", children=[inv])
        b.set_roots([itr])
        return b.finish(validate=False)

    def test_validate_good_teir(self) -> None:
        """A well-formed IR passes validation."""
        teir = self._good_teir()
        validate(teir)

    def test_guard_referencing_non_ancestor_axis(self) -> None:
        """Guard scoping is enforced."""
        b = TeirBuilder()
        b.add_tensor("in0", dtype="f32")
        b.add_tensor("out", dtype="f32")
        b.add_axis("a", extent=8, strides_by_tensor={"in0": 4, "out": 4})
        b.add_axis("b", extent=4, strides_by_tensor={"in0": 32, "out": 32})
        b.add_primitive(
            "copy",
            operation="Copy",
            axes={"M": [], "N": []},
            metadata={"data_type": "f32"},
        )
        # An invocation guarded by axis 'b', placed under an iteration over
        # axis 'a'. Axis 'b' is not iterated by any ancestor.
        inv = b.add_invocation("inv", primitive="copy", guard=guard(First("b")))
        it = b.add_iteration("it_a", axis="a", children=[inv])
        b.set_roots([it])
        with pytest.raises(
            etops.TeirValidationError,
            match="not iterated by any ancestor",
        ):
            b.finish(validate=True)

    def test_validate_aggregates_diagnostics(self) -> None:
        """Validation collects every violation in `diagnostics`."""
        # Construct an intentionally broken Teir directly to exercise
        # diagnostic accumulation. Multiple iteration nodes claim the same
        # invocation as their child; this yields two distinct errors.
        b = TeirBuilder()
        b.add_tensor("in0", dtype="f32")
        b.add_tensor("out", dtype="f32")
        b.add_axis("a", extent=8, strides_by_tensor={"in0": 4, "out": 4})
        b.add_axis("b", extent=4, strides_by_tensor={"in0": 32, "out": 32})
        b.add_primitive(
            "copy",
            operation="Copy",
            axes={"M": [], "N": []},
            metadata={"data_type": "f32"},
        )
        inv = b.add_invocation("inv", primitive="copy")
        it_a = b.add_iteration("it_a", axis="a", children=[inv])
        # Reuse the same invocation as another iteration's child — illegal.
        # The validator should report both "multiple parents" and the orphan
        # condition for it_b's missing root.
        it_b = b.add_iteration("it_b", axis="b", children=[inv])  # noqa: F841
        b.set_roots([it_a])
        with pytest.raises(etops.TeirValidationError) as info:
            b.finish(validate=True)
        # Three distinct violations: `inv` has two parents; `it_b` is
        # neither a root nor a child of any iteration node; and the walker
        # flags the detached `it_b` subtree.
        assert len(info.value.diagnostics) == 3

    def test_cursors_pre_order(self) -> None:
        """Pre-order walk visits roots before children."""
        teir = self._good_teir()
        ids = list(pre_order(teir))
        # The iteration node "it" is the root; "inv" is its child.
        assert ids == ["it", "inv"]


class TestMetadataSchema:
    """`Operation.metadata_schema` is closed by default; opt in via flag."""

    def _build_with_metadata(
        self,
        operation: str,
        metadata: dict[str, object],
    ) -> Teir:
        b = TeirBuilder()
        b.add_tensor("out", dtype="f32")
        b.add_axis("a", extent=4, strides_by_tensor={"out": 4})
        b.add_primitive(
            "p", operation=operation, axes={"M": [], "N": []}, metadata=metadata
        )
        inv = b.add_invocation("inv", primitive="p")
        root = b.add_iteration("iter_a", axis="a", children=[inv])
        b.set_roots([root])
        return b.finish(validate=True)

    def test_unknown_key_rejected_by_default(self) -> None:
        """A metadata key not in the schema is rejected."""

        with pytest.raises(
            etops.TeirEmissionError, match="not in the operation's schema"
        ):
            self._build_with_metadata("Zero", {"data_type": "f32", "tile": 32})


class TestParallelReductionRejection:
    """A reduction (``K``) axis cannot be iterated with policy=parallel."""

    def _build(self, k_policy: str) -> TeirBuilder:
        b = TeirBuilder().set_name("parallel_k")
        b.add_tensor("in0", dtype="f32")
        b.add_tensor("in1", dtype="f32")
        b.add_tensor("out", dtype="f32")
        # Free axes (m, n) on output, reduction axis k.
        b.add_axis("m", extent=4, strides_by_tensor={"in0": 4, "out": 4})
        b.add_axis("n", extent=4, strides_by_tensor={"in1": 4, "out": 16})
        b.add_axis("k", extent=4, strides_by_tensor={"in0": 16, "in1": 16})
        b.add_primitive(
            "ct",
            operation="Contraction",
            axes={"M": [], "N": [], "K": []},
            metadata={"data_type": "f32"},
        )
        b.add_primitive(
            "zero",
            operation="Zero",
            axes={"M": [], "N": []},
            metadata={"data_type": "f32"},
        )
        inv_ct = b.add_invocation("inv_ct", primitive="ct")
        inv_zero = b.add_invocation("inv_zero", primitive="zero")
        iter_k = b.add_iteration("iter_k", axis="k", children=[inv_ct], policy=k_policy)
        iter_n = b.add_iteration("iter_n", axis="n", children=[inv_zero, iter_k])
        root = b.add_iteration("iter_m", axis="m", children=[iter_n])
        b.set_roots([root])
        return b

    def test_sequential_k_validates(self) -> None:
        """Sequential iteration over the reduction axis passes validation."""

        teir = self._build("sequential").finish(validate=True)
        validate(teir)

    def test_parallel_k_rejected(self) -> None:
        """Parallel iteration over the reduction axis fails validation."""

        b = self._build("parallel")
        with pytest.raises(
            etops.TeirValidationError, match="zero stride on output tensor"
        ):
            b.finish(validate=True)


class TestRoleAxisStrideAlignment:
    """A primitive-consumed axis's byte stride must align to the element width."""

    def _build(self, stride_in0: int) -> TeirBuilder:
        b = TeirBuilder().set_name("role_align")
        b.add_tensor("in0", dtype="f32")
        b.add_tensor("out", dtype="f32")
        b.add_axis("a", extent=4, strides_by_tensor={"in0": 4, "out": 4})
        # `b` is a role axis on the Copy primitive — its byte stride must be
        # a multiple of the f32 element width (4).
        b.add_axis(
            "b",
            extent=2,
            strides_by_tensor={"in0": stride_in0, "out": 16},
        )
        b.add_primitive(
            "copy",
            operation="Copy",
            axes={"M": ["b"], "N": []},
            metadata={"data_type": "f32"},
        )
        inv = b.add_invocation("inv", primitive="copy")
        root = b.add_iteration("iter_a", axis="a", children=[inv])
        b.set_roots([root])
        return b

    def test_aligned_stride_passes(self) -> None:
        """A role-axis stride that is a multiple of the element width is OK."""

        teir = self._build(stride_in0=16).finish(validate=True)
        validate(teir)

    def test_misaligned_stride_rejected(self) -> None:
        """A role-axis stride that is not a multiple of the element width fails."""

        b = self._build(stride_in0=15)
        with pytest.raises(
            etops.TeirValidationError, match="not a multiple of the tensor"
        ):
            b.finish(validate=True)


class TestRoleAxisOffsets:
    """An axis consumed by a primitive role must have zero offsets everywhere."""

    def _build(self, axis_offset: int) -> TeirBuilder:
        b = TeirBuilder().set_name("role_offset")
        b.add_tensor("in0", dtype="f32")
        b.add_tensor("out", dtype="f32")
        # `a` is iterated by the schedule.
        b.add_axis("a", extent=4, strides_by_tensor={"in0": 4, "out": 4})
        # `b` is consumed by the primitive's M role. It carries `axis_offset`
        # on `in0`, which is illegal whenever the offset is non-zero.
        offsets = {"in0": axis_offset} if axis_offset != 0 else None
        b.add_axis(
            "b",
            extent=2,
            strides_by_tensor={"in0": 16, "out": 16},
            offsets_by_tensor=offsets,
        )
        b.add_primitive(
            "copy",
            operation="Copy",
            axes={"M": ["b"], "N": []},
            metadata={"data_type": "f32"},
        )
        inv = b.add_invocation("inv", primitive="copy")
        root = b.add_iteration("iter_a", axis="a", children=[inv])
        b.set_roots([root])
        return b

    def test_zero_offset_role_axis_passes(self) -> None:
        """A role axis with zero offsets validates."""

        teir = self._build(axis_offset=0).finish(validate=True)
        validate(teir)

    def test_non_zero_offset_role_axis_rejected(self) -> None:
        """A role axis with a non-zero offset fails validation."""

        b = self._build(axis_offset=8)
        with pytest.raises(
            etops.TeirValidationError,
            match="consumed by primitive 'copy' role 'M'",
        ):
            b.finish(validate=True)
