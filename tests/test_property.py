"""Hypothesis-driven property tests for the IR foundation and NumPy oracle.

Per-test ``@settings(max_examples=...)`` decorators are intentionally
absent so each test inherits the active profile selected in
``tests/conftest.py`` (``dev``, ``ci``, or ``nightly``).
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import assume, given
from hypothesis import strategies as st

import etops
from etops.diag import TeirLoweringError
from etops.ir import First, Last, Teir, guard, validate
from etops.ir.dtypes import resolve_numpy_dtype
from etops.textir import dump, parse
from etops.transforms import (
    canonicalize_ids,
    fuse_iterations,
    split_iteration,
)
from tests._helpers import backend_available, backend_param, tensor_axes_in_storage_order
from tests.oracles.numpy_oracle import NumpyOptimizationProfile, numpy_pipeline
from tests.strategies import binary_einsum_teir, unary_einsum_teir, unary_relu_teir

#: Production backends (everything except the numpy oracle) whose runtime is
#: actually built in this process. Computed once at import time.
_PRODUCTION_BACKENDS: tuple[str, ...] = tuple(
    b for b in etops.list_backends() if b != "numpy" and backend_available(b)
)


def _allocate_and_reference(
    teir: Teir, seed: int
) -> tuple[tuple[np.ndarray, np.ndarray, np.ndarray], np.ndarray]:
    """Allocate inputs and compute the np.einsum reference for a binary contraction.

    Returns ``((in0, in1, out), reference_for_out)``. ``out`` is zero-filled
    so the backend's accumulation can be checked against ``reference``.
    """

    tensor_ids = teir.tensor_ids
    in0_id, in1_id, out_id = tensor_ids[0], tensor_ids[1], tensor_ids[-1]

    in0_axes = tensor_axes_in_storage_order(teir, in0_id)
    in1_axes = tensor_axes_in_storage_order(teir, in1_id)
    out_axes = tensor_axes_in_storage_order(teir, out_id)

    np_dtype = resolve_numpy_dtype(teir.tensors[in0_id].dtype.name)
    rng = np.random.default_rng(seed)
    in0 = rng.standard_normal(tuple(teir.axes[a].extent for a in in0_axes)).astype(
        np_dtype
    )
    in1 = rng.standard_normal(tuple(teir.axes[a].extent for a in in1_axes)).astype(
        np_dtype
    )
    out = np.zeros(tuple(teir.axes[a].extent for a in out_axes), dtype=np_dtype)

    expr = f"{''.join(in0_axes)},{''.join(in1_axes)}->{''.join(out_axes)}"
    reference = np.einsum(expr, in0, in1, optimize=False)
    return (in0, in1, out), reference


def _allocate_relu_inputs(
    teir: Teir, seed: int
) -> tuple[tuple[np.ndarray, np.ndarray], np.ndarray]:
    """Allocate input + reference for a unary ReLU IR.

    Returns ``((in_arr, out_arr), reference)`` where ``in_arr`` carries
    mixed-sign normal random values so the clamp is exercised, ``out_arr``
    is zero-filled, and ``reference`` is the input permuted into the
    output's storage order with ``np.maximum(., 0)`` applied.
    """

    tensor_ids = teir.tensor_ids
    in_id, out_id = tensor_ids[0], tensor_ids[-1]
    in_axes = tensor_axes_in_storage_order(teir, in_id)
    out_axes = tensor_axes_in_storage_order(teir, out_id)
    np_dtype = resolve_numpy_dtype(teir.tensors[in_id].dtype.name)
    rng = np.random.default_rng(seed)
    in_arr = rng.standard_normal(tuple(teir.axes[a].extent for a in in_axes)).astype(
        np_dtype
    )
    out_arr = np.zeros(tuple(teir.axes[a].extent for a in out_axes), dtype=np_dtype)
    expr = f"{''.join(in_axes)}->{''.join(out_axes)}"
    reference = np.maximum(np.einsum(expr, in_arr, optimize=False), 0).astype(np_dtype)
    return (in_arr, out_arr), reference


@pytest.mark.property
class TestRoundTripProperty:
    """Every emitted `Teir` round-trips through textual IR."""

    @given(teir=unary_einsum_teir())
    def test_unary_round_trips(self, teir) -> None:
        """Unary einsum emitter output round-trips."""

        round_tripped = parse(dump(teir))
        assert round_tripped == teir
        # Mappings compare order-insensitive; lock the iteration order too.
        assert list(round_tripped.tensors.keys()) == list(teir.tensors.keys())
        assert list(round_tripped.axes.keys()) == list(teir.axes.keys())
        assert list(round_tripped.schedule.iterations.keys()) == list(
            teir.schedule.iterations.keys()
        )

    @given(teir=binary_einsum_teir())
    def test_binary_round_trips(self, teir) -> None:
        """Binary einsum emitter output round-trips."""

        round_tripped = parse(dump(teir))
        assert round_tripped == teir
        assert list(round_tripped.tensors.keys()) == list(teir.tensors.keys())
        assert list(round_tripped.axes.keys()) == list(teir.axes.keys())
        assert list(round_tripped.schedule.invocations.keys()) == list(
            teir.schedule.invocations.keys()
        )


@pytest.mark.property
class TestValidateProperty:
    """Every emitted `Teir` passes validation."""

    @given(teir=unary_einsum_teir())
    def test_unary_validates(self, teir) -> None:
        """Emitter never produces an invalid unary `Teir`."""

        validate(teir)

    @given(teir=binary_einsum_teir())
    def test_binary_validates(self, teir) -> None:
        """Emitter never produces an invalid binary `Teir`."""

        validate(teir)


@pytest.mark.property
class TestCanonicalizeIdempotent:
    """`canonicalize_ids` is idempotent."""

    @given(teir=unary_einsum_teir(max_rank=3))
    def test_unary_idempotent(self, teir) -> None:
        once = canonicalize_ids(teir)
        twice = canonicalize_ids(once)
        assert once == twice

    @given(teir=binary_einsum_teir(max_rank=2))
    def test_binary_idempotent(self, teir) -> None:
        once = canonicalize_ids(teir)
        twice = canonicalize_ids(once)
        assert once == twice


@pytest.mark.property
class TestNumpyPipelineIdempotent:
    """Running the numpy pipeline twice produces an equal Teir."""

    @given(teir=binary_einsum_teir(max_rank=2))
    def test_pipeline_idempotent(self, teir) -> None:
        pipeline = numpy_pipeline()
        profile = NumpyOptimizationProfile()
        once = pipeline.run(teir, profile=profile)
        twice = pipeline.run(once, profile=profile)
        assert once == twice


@pytest.mark.property
class TestSplitFuseInverse:
    """`fuse_iterations(split_iteration(t, node_id, n), node_id)` returns to
    a semantically equivalent IR."""

    @given(teir=unary_einsum_teir(max_rank=2))
    def test_round_trip(self, teir) -> None:
        # Pick the first axis whose extent has a non-trivial divisor.
        target_axis: str | None = None
        target_inner: int = 0
        for aid, axis in teir.axes.items():
            for d in range(2, axis.extent):
                if axis.extent % d == 0:
                    target_axis = aid
                    target_inner = d
                    break
            if target_axis is not None:
                break
        assume(target_axis is not None)
        node_id = next(
            nid
            for nid, it in teir.schedule.iterations.items()
            if it.axis == target_axis
        )
        split = split_iteration(teir, node_id, inner_extent=target_inner)
        fused = fuse_iterations(split, node_id)
        # The fused IR's set of iteration-active axis extents matches the
        # original. Schedule-centric split/fuse leaves orphan axes behind
        # in `axes` until `DropUnusedAxes` runs; compare only the axes
        # still iterated by the schedule.
        active = {it.axis for it in fused.schedule.iterations.values()}
        fused_extents = sorted(fused.axes[a].extent for a in active)
        original_extents = sorted(a.extent for a in teir.axes.values())
        assert fused_extents == original_extents


@pytest.mark.property
class TestMultiTermGuardRoundTrip:
    """Multi-term guards round-trip through the textual IR."""

    def test_conjunctive_guard_round_trips(self) -> None:
        """A schedule with a ``first(@iter_a) & last(@iter_b)`` guard round-trips."""

        from etops.ir import TeirBuilder

        b = TeirBuilder().set_name("multi_guard")
        b.add_tensor("out", dtype="f32")
        b.add_axis("a", extent=4, strides_by_tensor={"out": 16})
        b.add_axis("b", extent=4, strides_by_tensor={"out": 4})
        b.add_primitive(
            "zero",
            operation="Zero",
            axes={"M": [], "N": []},
            metadata={"data_type": "f32"},
        )
        inv = b.add_invocation(
            "inv_zero",
            primitive="zero",
            guard=guard(First("iter_a"), Last("iter_b")),
        )
        iter_b = b.add_iteration("iter_b", axis="b", children=[inv])
        iter_a = b.add_iteration("iter_a", axis="a", children=[iter_b])
        b.set_roots([iter_a])
        teir = b.finish(validate=True)
        assert parse(dump(teir)) == teir


@pytest.mark.property
@pytest.mark.skipif(
    not _PRODUCTION_BACKENDS, reason="no production backends registered"
)
class TestBackendsAgreeWithOracle:
    """After optimization, every registered production backend produces
    oracle-matching output for an arbitrary binary contraction.

    Architectural invariant under test: the optimization pipeline normalizes
    any einsum into a backend-dispatchable shape, so
    ``etops.compile(teir, backend=...)`` must succeed for every IR the
    property strategy produces. A ``TeirLoweringError`` here indicates an
    optimizer regression and is allowed to fail loudly rather than being
    swallowed.
    """

    @backend_param(*_PRODUCTION_BACKENDS)
    @given(teir=binary_einsum_teir(), seed=st.integers(min_value=0, max_value=2**32 - 1))
    def test_contraction_matches_numpy(
        self, teir: Teir, seed: int, backend: str
    ) -> None:
        inputs, reference = _allocate_and_reference(teir, seed=seed)
        op = etops.compile(teir, backend=backend)
        op.execute(*inputs)
        np.testing.assert_allclose(inputs[-1], reference, atol=1e-4, rtol=1e-4)


@pytest.mark.property
@pytest.mark.skipif(
    not _PRODUCTION_BACKENDS, reason="no production backends registered"
)
class TestReluAgreesWithOracle:
    """Every registered production backend either applies ReLU correctly
    or refuses at compile time. ReLU is exact (compare + select; no float
    arithmetic), so equality is bit-exact under any layout the strategy
    emits.

    Some (backend, dtype) combinations are documented refusals — e.g. the
    TPP backend declines f64 ReLU because libxsmm has no JIT'd f64 RELU
    kernel and its C reference fallback mishandles f64 (see the
    ``is_reference_kernel`` guard in ``compile_unary``). The test treats a
    ``TeirLoweringError`` as a valid contract outcome, distinct from
    silently-wrong output.
    """

    @backend_param(*_PRODUCTION_BACKENDS)
    @given(
        teir=unary_relu_teir(dtype="f32"),
        seed=st.integers(min_value=0, max_value=2**32 - 1),
    )
    def test_relu_f32_matches_numpy(
        self, teir: Teir, seed: int, backend: str
    ) -> None:
        inputs, reference = _allocate_relu_inputs(teir, seed=seed)
        op = etops.compile(teir, backend=backend)
        op.execute(*inputs)
        np.testing.assert_array_equal(inputs[-1], reference)

    @backend_param(*_PRODUCTION_BACKENDS)
    @given(
        teir=unary_relu_teir(dtype="f64"),
        seed=st.integers(min_value=0, max_value=2**32 - 1),
    )
    def test_relu_f64_matches_numpy(
        self, teir: Teir, seed: int, backend: str
    ) -> None:
        inputs, reference = _allocate_relu_inputs(teir, seed=seed)
        try:
            op = etops.compile(teir, backend=backend)
        except TeirLoweringError:
            # Documented refusal (e.g. TPP f64). Refusal is contract;
            # silently-wrong output would be the bug.
            return
        op.execute(*inputs)
        np.testing.assert_array_equal(inputs[-1], reference)
