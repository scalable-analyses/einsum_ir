"""
TileIR function builder for the tileir backend.

Constructs a ``cuda.tile._ir.ir.Function`` object graph from a
:class:`ConfigAnalysis`.  The resulting IR can be passed through the
standard TileIR optimization passes and then serialised to bytecode.

The builder supports:
  * Shared (grid-mapped) loops
  * Sequential for-loops (non-K and K)
  * MMA primitives (GEMM / BRGEMM)
  * ``prim_first`` = ``zero`` (beta=0) or ``none`` (beta=1)
  * ``prim_last``  = ``relu`` or ``none``
  * All five data types (FP32, FP64, FP16, BF16, TF32)

Each tensor has its own view containing only the dimensions where its
stride is non-zero.  The index tuples for ``TileLoad``/``TileStore``
are built per-tensor from the ``index_map_*`` analysis fields.
"""

from __future__ import annotations

__all__ = ["build_function"]

from typing import Dict, Optional, Tuple

from etops.types import PrimType

from etops.backends._tileir.config_analysis import ConfigAnalysis, IndexMapEntry


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def build_function(analysis: ConfigAnalysis, kernel_name: str) -> object:
    """Build a TileIR ``Function`` from *analysis*.

    Args:
        analysis: Frozen config analysis produced by
            :func:`~etops.backends._tileir.config_analysis.analyze_config`.
        kernel_name: Name to assign to the generated kernel function.

    Returns:
        A tuple ``(Function, IRContext, temp_dir)`` ready for optimization
        passes.
    """
    from cuda.tile._compile import _get_max_supported_bytecode_version
    from cuda.tile._cext import default_tile_context
    from cuda.tile._exception import Loc
    from cuda.tile._ir.ir import IRContext, Block, Function, Var
    from cuda.tile._ir.ops import (
        AssumeDivBy,
        Continue,
        Loop,
        MakeTensorView,
        RawBinaryArithmeticOperation,
        Return,
        TileAsType,
        TileBid,
        TileLoad,
        TileMma,
        TileReshape,
        TileStore,
        TypedConst,
    )
    from cuda.tile._ir.type import (
        ArrayTy,
        PointerTy,
        SizeTy,
        TupleTy,
        make_tile_ty,
    )
    from cuda.tile._ir.typing_support import datatype
    from cuda.tile import PaddingMode

    import tempfile

    _LOC = Loc(line=0, col=0)

    # Resolve data types from attribute names
    storage_dt = getattr(datatype, analysis.storage_dtype_name)
    acc_dt = getattr(datatype, analysis.acc_dtype_name)

    # -----------------------------------------------------------------------
    # Helper closures (capture ctx, _LOC, types)
    # -----------------------------------------------------------------------

    temp_dir = tempfile.mkdtemp(prefix="tileir_")
    bytecode_version = _get_max_supported_bytecode_version(temp_dir)
    ctx = IRContext(
        config=default_tile_context.config,
        tileiras_version=bytecode_version,
    )

    def _const(block: Block, value: int) -> Var:
        """Emit ``TypedConst i32`` and return its ``Var``."""
        v = ctx.make_temp(_LOC)
        v.set_type(make_tile_ty(datatype.int32, ()))
        v.set_constant(value)
        block.append(TypedConst(result_vars=(v,), loc=_LOC, value=value))
        return v

    def _arith(block: Block, fn: str, lhs: Var, rhs: Var) -> Var:
        """Emit ``RawBinaryArithmeticOperation`` and return the result."""
        v = ctx.make_temp(_LOC)
        v.set_type(make_tile_ty(datatype.int32, ()))
        block.append(
            RawBinaryArithmeticOperation(
                result_vars=(v,),
                loc=_LOC,
                fn=fn,
                rounding_mode=None,
                flush_to_zero=False,
                lhs=lhs,
                rhs=rhs,
            )
        )
        return v

    def _arith_float(
        block: Block, fn: str, lhs: Var, rhs: Var, result_type: object
    ) -> Var:
        """Emit a float ``RawBinaryArithmeticOperation``."""
        v = ctx.make_temp(_LOC)
        v.set_type(result_type)
        block.append(
            RawBinaryArithmeticOperation(
                result_vars=(v,),
                loc=_LOC,
                fn=fn,
                rounding_mode=None,
                flush_to_zero=False,
                lhs=lhs,
                rhs=rhs,
            )
        )
        return v

    def _make_array_ty(
        shape: Tuple[int, ...],
        strides: Tuple[int, ...],
        elem_dtype: object,
    ) -> ArrayTy:
        """Build a static ``ArrayTy`` with optimal alignment hints."""
        return ArrayTy(
            elem_dtype,
            shape=TupleTy([SizeTy(s) for s in shape]),
            strides=TupleTy([SizeTy(s) for s in strides]),
            elements_disjoint=True,
            base_ptr_div_by=16,
            stride_div_by=tuple(None for _ in strides),
            shape_div_by=tuple(None for _ in shape),
        )

    def _ptr_setup(
        block: Block,
        ptr_var: Var,
        shape: Tuple[int, ...],
        strides: Tuple[int, ...],
        elem_dtype: object,
    ) -> Var:
        """``AssumeDivBy(16)`` + ``MakeTensorView`` → tensor-view ``Var``."""
        aligned = ctx.make_temp(_LOC)
        aligned.set_type(make_tile_ty(PointerTy(elem_dtype), ()))
        block.append(
            AssumeDivBy(result_vars=(aligned,), loc=_LOC, divisor=16, x=ptr_var)
        )

        tv = ctx.make_temp(_LOC)
        tv.set_type(_make_array_ty(shape, strides, elem_dtype))
        block.append(
            MakeTensorView(
                result_vars=(tv,),
                loc=_LOC,
                base_ptr=aligned,
                shape=(),
                dynamic_strides=(),
            )
        )
        return tv

    # -----------------------------------------------------------------------
    # Root block & ABI parameters
    # -----------------------------------------------------------------------
    root = Block(ctx, _LOC)

    # Only 3 pointer parameters — no shape/stride ABI slots.
    def _param_ptr(name: str) -> Var:
        v = ctx.make_var(name, _LOC)
        v.set_type(make_tile_ty(PointerTy(storage_dt), ()))
        return v

    in0_ptr = _param_ptr("in0")
    in1_ptr = _param_ptr("in1")
    out_ptr = _param_ptr("out")
    root.params = (in0_ptr, in1_ptr, out_ptr)

    # -----------------------------------------------------------------------
    # Per-tensor views (each with only non-zero-stride dims)
    # -----------------------------------------------------------------------
    tv_in0 = _ptr_setup(
        root,
        in0_ptr,
        analysis.tensor_shape_in0,
        analysis.tensor_strides_in0,
        storage_dt,
    )
    tv_in1 = _ptr_setup(
        root,
        in1_ptr,
        analysis.tensor_shape_in1,
        analysis.tensor_strides_in1,
        storage_dt,
    )
    tv_out = _ptr_setup(
        root,
        out_ptr,
        analysis.tensor_shape_out,
        analysis.tensor_strides_out,
        storage_dt,
    )

    # -----------------------------------------------------------------------
    # Block-index decomposition for shared loops
    # -----------------------------------------------------------------------
    num_shared = len(analysis.shared_loop_ids)

    shared_idx_vars: Dict[int, Var] = {}  # shared_pos → Var

    if num_shared > 0:
        bid = ctx.make_temp(_LOC)
        bid.set_type(make_tile_ty(datatype.int32, ()))
        root.append(TileBid(result_vars=(bid,), loc=_LOC, axis=0))

        # Decompose bid into per-dimension indices using row-major strides.
        remaining = bid
        for pos in range(num_shared):
            stride_val = analysis.shared_loop_strides[pos]
            if stride_val == 1:
                # Last shared dim: remaining IS the index
                shared_idx_vars[pos] = remaining
            else:
                stride_c = _const(root, stride_val)
                idx = _arith(root, "floordiv", remaining, stride_c)
                shared_idx_vars[pos] = idx
                # remaining = remaining - idx * stride
                prod = _arith(root, "mul", idx, _const(root, stride_val))
                remaining = _arith(root, "sub", remaining, prod)

    # -----------------------------------------------------------------------
    # Per-tensor index tuple builder
    # -----------------------------------------------------------------------

    def _build_tensor_index(
        block: Block,
        index_map: Tuple[IndexMapEntry, ...],
        seq_vars: Dict[int, Var],
    ) -> tuple:
        """Build an index tuple for a single tensor's view.

        Each position in the tensor's view maps to:
          - ``("shared", pos)`` → decomposed block-id variable
          - ``("seq", offset)`` → sequential loop induction variable
          - ``("prim", 0)``     → constant 0

        Args:
            block: The current IR block (for emitting ``c0`` constants).
            index_map: Per-position mapping from the analysis.
            seq_vars: Map from seq_loop offset → loop induction ``Var``.

        Returns:
            Tuple of ``Var`` objects, one per view position.
        """
        indices = []
        c0 = _const(block, 0)
        for kind, position in index_map:
            if kind == "shared":
                indices.append(shared_idx_vars[position])
            elif kind == "seq":
                indices.append(seq_vars.get(position, c0))
            else:
                # prim → constant 0
                indices.append(c0)
        return tuple(indices)

    # -----------------------------------------------------------------------
    # Build the contraction body
    # -----------------------------------------------------------------------
    #
    # Structure (from outer to inner):
    #   1. Shared dims  → decomposed from bid (above)
    #   2. Seq non-K dims → for-loops
    #   3. prim_first  → accumulator init (zero or load)
    #   4. Seq K dims   → for-loop (K reduction)
    #       4a. Load A (in0), Load B (in1)
    #       4b. MMA: acc = mma(B, A, acc)
    #   5. prim_last   → optional relu
    #   6. Cast + Store result

    # -- Innermost body: K-loop body (or direct if no seq-K) ----------------

    def _build_k_body(
        block: Block,
        acc_in: Var,
        seq_vars: Dict[int, Var],
        k_var: Optional[Var],
        k_seq_offset: Optional[int],
    ) -> Var:
        """Build the MMA body: load A, load B, mma.

        Returns the new accumulator Var.
        """
        # Merge k_var into seq_vars for this iteration
        merged_seq: Dict[int, Var] = dict(seq_vars)
        if k_var is not None and k_seq_offset is not None:
            merged_seq[k_seq_offset] = k_var

        # Load in0 (A / y-operand) with per-tensor index
        idx_in0 = _build_tensor_index(block, analysis.index_map_in0, merged_seq)

        a_nd = ctx.make_temp(_LOC)
        a_nd.set_type(make_tile_ty(storage_dt, analysis.load_tile_shape_in0))
        block.append(
            TileLoad(
                result_vars=(a_nd,),
                loc=_LOC,
                order=analysis.tensor_order_in0,
                padding_mode=PaddingMode.ZERO,
                latency=None,
                allow_tma=True,
                array=tv_in0,
                index=idx_in0,
            )
        )

        # Reshape to 2-D for MMA: (K, M)
        a_2d = ctx.make_temp(_LOC)
        a_2d.set_type(make_tile_ty(storage_dt, analysis.mma_y_shape))
        block.append(TileReshape(result_vars=(a_2d,), loc=_LOC, x=a_nd))

        # Load in1 (B / x-operand) with per-tensor index
        idx_in1 = _build_tensor_index(block, analysis.index_map_in1, merged_seq)

        b_nd = ctx.make_temp(_LOC)
        b_nd.set_type(make_tile_ty(storage_dt, analysis.load_tile_shape_in1))
        block.append(
            TileLoad(
                result_vars=(b_nd,),
                loc=_LOC,
                order=analysis.tensor_order_in1,
                padding_mode=PaddingMode.ZERO,
                latency=None,
                allow_tma=True,
                array=tv_in1,
                index=idx_in1,
            )
        )

        # Reshape to 2-D for MMA: (N, K)
        b_2d = ctx.make_temp(_LOC)
        b_2d.set_type(make_tile_ty(storage_dt, analysis.mma_x_shape))
        block.append(TileReshape(result_vars=(b_2d,), loc=_LOC, x=b_nd))

        # MMA: acc_new = mma(x=B, y=A, acc)
        acc_new = ctx.make_temp(_LOC)
        acc_new.set_type(make_tile_ty(acc_dt, analysis.acc_shape))
        block.append(
            TileMma(
                result_vars=(acc_new,),
                loc=_LOC,
                x=b_2d,
                y=a_2d,
                acc=acc_in,
            )
        )
        return acc_new

    def _wrap_k_loops(
        block: Block,
        acc_init: Var,
        seq_vars: Dict[int, Var],
    ) -> Var:
        """Wrap the K-loop(s) around the MMA body.

        If there are multiple seq-K dims, they are nested from outer to
        inner (matching the config order).

        Returns the final accumulator after all K iterations.
        """
        seq_k_offsets = []
        for kid in analysis.seq_k_loop_ids:
            # offset within the seq portion of the view
            seq_offset = list(analysis.seq_loop_ids).index(kid)
            seq_k_offsets.append(seq_offset)

        if len(seq_k_offsets) == 0:
            # No seq-K: single MMA application
            return _build_k_body(block, acc_init, seq_vars, None, None)

        # Build nested K loops from outermost to innermost
        return _build_nested_k_loops(block, acc_init, seq_vars, seq_k_offsets, 0)

    def _build_nested_k_loops(
        block: Block,
        acc_carry: Var,
        seq_vars: Dict[int, Var],
        seq_k_offsets: list,
        depth: int,
    ) -> Var:
        """Recursively build nested K-loops."""
        if depth >= len(seq_k_offsets):
            # Base case: innermost — do the MMA
            return _build_k_body(block, acc_carry, seq_vars, None, None)

        seq_offset = seq_k_offsets[depth]
        config_idx = analysis.seq_k_loop_ids[depth]
        loop_size = analysis.dim_sizes[config_idx]

        lb = _const(block, 0)
        ub = _const(block, loop_size)
        step = _const(block, 1)

        body = Block(ctx, _LOC)
        k_iter = ctx.make_var(f"k{depth}", _LOC)
        k_iter.set_type(make_tile_ty(datatype.int32, ()))
        acc_param = ctx.make_var(f"acc_k{depth}", _LOC)
        acc_param.set_type(make_tile_ty(acc_dt, analysis.acc_shape))
        body.params = (k_iter, acc_param)

        # Set the current K dim in seq_vars
        inner_seq: Dict[int, Var] = dict(seq_vars)
        inner_seq[seq_offset] = k_iter

        if depth + 1 < len(seq_k_offsets):
            # More K loops to nest
            acc_result = _build_nested_k_loops(
                body, acc_param, inner_seq, seq_k_offsets, depth + 1
            )
        else:
            # Innermost K loop — do MMA
            acc_result = _build_k_body(body, acc_param, inner_seq, k_iter, seq_offset)

        body.append(Continue(result_vars=(), loc=_LOC, values=(acc_result,)))

        loop_out = ctx.make_temp(_LOC)
        loop_out.set_type(make_tile_ty(acc_dt, analysis.acc_shape))
        block.append(
            Loop(
                result_vars=(loop_out,),
                loc=_LOC,
                start=lb,
                stop=ub,
                step=step,
                initial_values=(acc_carry,),
                body=body,
            )
        )
        return loop_out

    def _build_contraction_at_point(
        block: Block,
        seq_vars: Dict[int, Var],
    ) -> None:
        """Build contraction + store for a single shared/seq point.

        Handles prim_first (acc init), K-loops, prim_last (relu), cast,
        and store.
        """
        # -- Accumulator init -------------------------------------------------
        if analysis.prim_first == PrimType.zero:
            # Zero-init accumulator
            acc_init = ctx.make_temp(_LOC)
            acc_init.set_type(make_tile_ty(acc_dt, analysis.acc_shape))
            acc_init.set_constant(0)
            block.append(TypedConst(result_vars=(acc_init,), loc=_LOC, value=0))
        else:
            # prim_first == none → beta=1: load existing output as acc
            idx_out = _build_tensor_index(block, analysis.index_map_out, seq_vars)

            out_nd = ctx.make_temp(_LOC)
            out_nd.set_type(make_tile_ty(storage_dt, analysis.store_tile_shape_out))
            block.append(
                TileLoad(
                    result_vars=(out_nd,),
                    loc=_LOC,
                    order=analysis.tensor_order_out,
                    padding_mode=PaddingMode.ZERO,
                    latency=None,
                    allow_tma=True,
                    array=tv_out,
                    index=idx_out,
                )
            )
            # Reshape to 2-D acc shape
            out_2d = ctx.make_temp(_LOC)
            out_2d.set_type(make_tile_ty(storage_dt, analysis.acc_shape))
            block.append(TileReshape(result_vars=(out_2d,), loc=_LOC, x=out_nd))
            # Cast to acc dtype if needed
            if analysis.storage_dtype_name != analysis.acc_dtype_name:
                acc_init = ctx.make_temp(_LOC)
                acc_init.set_type(make_tile_ty(acc_dt, analysis.acc_shape))
                block.append(TileAsType(result_vars=(acc_init,), loc=_LOC, x=out_2d))
            else:
                acc_init = out_2d

        # -- K reduction loops ------------------------------------------------
        acc_result = _wrap_k_loops(block, acc_init, seq_vars)

        # -- prim_last: optional ReLU -----------------------------------------
        if analysis.prim_last == PrimType.relu:
            zero_tile = ctx.make_temp(_LOC)
            zero_tile.set_type(make_tile_ty(acc_dt, analysis.acc_shape))
            zero_tile.set_constant(0)
            block.append(TypedConst(result_vars=(zero_tile,), loc=_LOC, value=0))
            acc_result = _arith_float(
                block,
                "max",
                acc_result,
                zero_tile,
                make_tile_ty(acc_dt, analysis.acc_shape),
            )

        # -- Cast acc → storage dtype + reshape + store -----------------------
        if analysis.storage_dtype_name != analysis.acc_dtype_name:
            r_storage = ctx.make_temp(_LOC)
            r_storage.set_type(make_tile_ty(storage_dt, analysis.acc_shape))
            block.append(TileAsType(result_vars=(r_storage,), loc=_LOC, x=acc_result))
        else:
            r_storage = acc_result

        # Reshape from 2-D (N, M) to per-tensor N-D store tile
        r_nd = ctx.make_temp(_LOC)
        r_nd.set_type(make_tile_ty(storage_dt, analysis.store_tile_shape_out))
        block.append(TileReshape(result_vars=(r_nd,), loc=_LOC, x=r_storage))

        idx_out = _build_tensor_index(block, analysis.index_map_out, seq_vars)
        block.append(
            TileStore(
                result_vars=(),
                loc=_LOC,
                order=analysis.tensor_order_out,
                latency=None,
                allow_tma=None,
                array=tv_out,
                index=idx_out,
                tile=r_nd,
            )
        )

    # -----------------------------------------------------------------------
    # Wrap seq non-K loops around the contraction
    # -----------------------------------------------------------------------

    def _wrap_seq_non_k_loops(
        block: Block,
        non_k_offsets: list,
        depth: int,
        seq_vars: Dict[int, Var],
    ) -> None:
        """Recursively wrap sequential non-K loops.

        At the base case (all non-K loops emitted), emit the full
        contraction body.
        """
        if depth >= len(non_k_offsets):
            _build_contraction_at_point(block, seq_vars)
            return

        seq_offset = non_k_offsets[depth]
        config_idx = analysis.seq_non_k_loop_ids[depth]
        loop_size = analysis.dim_sizes[config_idx]

        lb = _const(block, 0)
        ub = _const(block, loop_size)
        step = _const(block, 1)

        body = Block(ctx, _LOC)
        loop_var = ctx.make_var(f"seq{depth}", _LOC)
        loop_var.set_type(make_tile_ty(datatype.int32, ()))
        body.params = (loop_var,)

        inner_seq: Dict[int, Var] = dict(seq_vars)
        inner_seq[seq_offset] = loop_var

        _wrap_seq_non_k_loops(body, non_k_offsets, depth + 1, inner_seq)

        body.append(Continue(result_vars=(), loc=_LOC, values=()))

        block.append(
            Loop(
                result_vars=(),
                loc=_LOC,
                start=lb,
                stop=ub,
                step=step,
                initial_values=(),
                body=body,
            )
        )

    # Compute non-K seq offsets (position within the seq portion of view)
    non_k_offsets = []
    for nk_id in analysis.seq_non_k_loop_ids:
        seq_offset = list(analysis.seq_loop_ids).index(nk_id)
        non_k_offsets.append(seq_offset)

    initial_seq_vars: Dict[int, Var] = {}
    _wrap_seq_non_k_loops(root, non_k_offsets, 0, initial_seq_vars)

    # -----------------------------------------------------------------------
    # Return
    # -----------------------------------------------------------------------
    root.append(Return(result_vars=(), loc=_LOC))
    func = Function(body=root, name=kernel_name, loc=_LOC)
    return func, ctx, temp_dir
