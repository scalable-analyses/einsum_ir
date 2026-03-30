"""
Compiler pipeline for the tileir backend.

Takes a TileIR ``Function`` (produced by :mod:`ir_builder`), applies the
standard 8-pass optimization pipeline, serialises to bytecode, compiles to
a ``.cubin``, and wraps the result in a CuPy ``RawKernel`` for launch.

Includes:
  * In-memory SHA-256 kernel cache (thread-safe).
  * ``ETOPS_DUMP_IR`` environment-variable controlled IR / SASS text dump.
"""

from __future__ import annotations

__all__ = ["compile_binary_analysis", "compile_unary_analysis", "TileIRKernel"]

import hashlib
import logging
import os
import shutil
import subprocess
import sys
import threading
from pathlib import Path
from typing import Dict, FrozenSet

from etops.backends._tileir.config_analysis import (
    BinaryConfigAnalysis,
    UnaryConfigAnalysis,
)

_logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dump configuration (ETOPS_DUMP_IR)
# ---------------------------------------------------------------------------

_DUMP_ENV = "ETOPS_DUMP_IR"

#: Valid tokens recognised in the comma-separated ``ETOPS_DUMP_IR`` value.
_VALID_TOKENS = frozenset({"tileir_before", "tileir_after", "tileir_all", "sass"})


def _dump_tokens() -> FrozenSet[str]:
    """Return the set of active dump tokens from ``ETOPS_DUMP_IR``.

    The environment variable is a comma-separated list of tokens
    (case-insensitive, whitespace-tolerant).  Valid tokens:

    * ``tileir_before`` -- dump TileIR text before optimisation passes.
    * ``tileir_after``  -- dump TileIR text after all optimisation passes.
    * ``tileir_all``    -- dump TileIR text before, after each individual
      pass, and after all passes.
    * ``sass``          -- dump SASS disassembly of the compiled cubin.

    Unrecognised tokens are silently ignored so that forward-compatible
    values do not break older releases.
    """
    raw = os.environ.get(_DUMP_ENV, "").strip().lower()
    if not raw:
        return frozenset()
    tokens = frozenset(t.strip() for t in raw.split(",") if t.strip())
    return tokens & _VALID_TOKENS


def _dump_ir(label: str, func_ir: object) -> None:
    """Print IR text to stderr with a header *label*."""
    sep = "=" * 72
    sys.stderr.write(f"{sep}\nTileIR  *** {label} ***\n{sep}\n")
    sys.stderr.write(func_ir.body.to_string())
    sys.stderr.write("\n\n")
    sys.stderr.flush()


def _dump_sass(cubin_path: str) -> None:
    """Disassemble *cubin_path* with ``cuobjdump -sass`` and write to stderr.

    Raises:
        RuntimeError: If ``cuobjdump`` is not found on ``PATH``.
    """
    cuobjdump = shutil.which("cuobjdump")
    if cuobjdump is None:
        raise RuntimeError(
            "SASS dump requested (ETOPS_DUMP_IR contains 'sass') but "
            "'cuobjdump' was not found on PATH.  Install the CUDA Toolkit "
            "or ensure 'cuobjdump' is accessible."
        )

    result = subprocess.run(
        [cuobjdump, "-sass", cubin_path],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"cuobjdump -sass failed (exit code {result.returncode}):\n{result.stderr}"
        )

    sep = "=" * 72
    sys.stderr.write(f"{sep}\nSASS  *** {cubin_path} ***\n{sep}\n")
    sys.stderr.write(result.stdout)
    sys.stderr.write("\n\n")
    sys.stderr.flush()


# ---------------------------------------------------------------------------
# Pass pipeline
# ---------------------------------------------------------------------------

_PASS_NAMES = [
    "eliminate_assign_ops",
    "dead_code_elimination_pass",
    "alias_analysis_pass",
    "token_order_pass",
    "rewrite_patterns",
    "hoist_loop_invariants",
    "split_loops",
    "dead_code_elimination_pass",
]


def _apply_passes(func_ir: object, tokens: FrozenSet[str]) -> None:
    """Apply the standard 8-pass IR transformation pipeline in-place."""
    from cuda.tile._passes.alias_analysis import alias_analysis_pass
    from cuda.tile._passes.code_motion import hoist_loop_invariants
    from cuda.tile._passes.dce import dead_code_elimination_pass
    from cuda.tile._passes.eliminate_assign_ops import eliminate_assign_ops
    from cuda.tile._passes.loop_split import split_loops
    from cuda.tile._passes.rewrite_patterns import rewrite_patterns
    from cuda.tile._passes.token_order import token_order_pass

    dump_all = "tileir_all" in tokens
    root = func_ir.body

    passes = [
        ("eliminate_assign_ops", lambda r: eliminate_assign_ops(r)),
        ("dead_code_elimination_pass", lambda r: dead_code_elimination_pass(r)),
        ("alias_analysis_pass", lambda r: alias_analysis_pass(r)),
        ("token_order_pass", lambda r, _aa=None: token_order_pass(r, _aa)),
        ("rewrite_patterns", lambda r: rewrite_patterns(r)),
        ("hoist_loop_invariants", lambda r: hoist_loop_invariants(r)),
        ("split_loops", lambda r: split_loops(r)),
        ("dead_code_elimination_pass (2)", lambda r: dead_code_elimination_pass(r)),
    ]

    alias_result = None
    for name, fn in passes:
        if name == "alias_analysis_pass":
            alias_result = alias_analysis_pass(root)
            if dump_all:
                _dump_ir(f"AFTER {name}", func_ir)
            continue
        if name == "token_order_pass":
            token_order_pass(root, alias_result)
            if dump_all:
                _dump_ir(f"AFTER {name}", func_ir)
            continue
        fn(root)
        if dump_all:
            _dump_ir(f"AFTER {name}", func_ir)


# ---------------------------------------------------------------------------
# Bytecode → cubin
# ---------------------------------------------------------------------------


def _build_cubin(
    func_ir: object,
    temp_dir: str,
    kernel_name: str,
) -> Path:
    """Serialise *func_ir* to bytecode and compile to a cubin.

    Returns:
        Path to the compiled ``.cubin`` file.
    """
    from cuda.tile._compile import (
        compile_cubin,
        get_sm_arch,
        _get_max_supported_bytecode_version,
    )
    from cuda.tile._compiler_options import CompilerOptions
    import cuda.tile._bytecode as bc
    from cuda.tile._ir2bytecode import generate_bytecode_for_kernel

    bytecode_version = _get_max_supported_bytecode_version(temp_dir)
    sm_arch = get_sm_arch()
    options = CompilerOptions()

    buf = bytearray()
    with bc.write_bytecode(
        num_functions=1, buf=buf, version=bytecode_version
    ) as writer:
        generate_bytecode_for_kernel(
            func_ir,
            options,
            sm_arch,
            writer,
            anonymize_debug_attr=True,
        )

    bc_path = Path(temp_dir) / f"{kernel_name}.tileirbc"
    bc_path.write_bytes(buf)

    cubin_path = compile_cubin(str(bc_path), options, sm_arch, timeout_sec=120)
    _logger.debug(
        "Compiled cubin: %s (%d bytes)", cubin_path, cubin_path.stat().st_size
    )
    return cubin_path


# ---------------------------------------------------------------------------
# Kernel wrapper (CuPy RawModule)
# ---------------------------------------------------------------------------


class TileIRKernel:
    """Compiled tileir kernel launchable via CuPy.

    Attributes:
        grid_size: Number of CUDA thread blocks.
        kernel_name: Name of the kernel entry point in the cubin.
        cubin_path: Filesystem path to the ``.cubin``.
        is_unary: Whether this is a unary kernel (2-pointer ABI).
    """

    def __init__(
        self,
        cubin_path: str,
        kernel_name: str,
        grid_size: int,
        is_unary: bool = False,
    ) -> None:
        self.cubin_path = cubin_path
        self.kernel_name = kernel_name
        self.grid_size = grid_size
        self.is_unary = is_unary

        import cupy as cp

        self._module = cp.RawModule(path=cubin_path)
        self._kernel = self._module.get_function(kernel_name)

    def __call__(self, in0: object, in1: object, out: object) -> None:
        """Launch the kernel.

        Args:
            in0: First input (CuPy or PyTorch CUDA tensor).
            in1: Second input (CuPy or PyTorch CUDA tensor).  Ignored
                for unary kernels.
            out: Output (CuPy or PyTorch CUDA tensor, pre-allocated).
        """
        import cupy as cp

        in0 = self._as_cupy(in0)
        out = self._as_cupy(out)

        stream = cp.cuda.get_current_stream()
        # Block dims are embedded in cubin metadata; pass (1,1,1).
        if self.is_unary:
            self._kernel(
                (self.grid_size,),
                (1, 1, 1),
                (in0, out),
                stream=stream,
            )
        else:
            in1 = self._as_cupy(in1)
            self._kernel(
                (self.grid_size,),
                (1, 1, 1),
                (in0, in1, out),
                stream=stream,
            )

    @staticmethod
    def _as_cupy(tensor: object) -> object:
        """Convert *tensor* to a CuPy ndarray if it is not already one.

        Accepts CuPy ndarrays (returned as-is) and any object that
        exposes ``__cuda_array_interface__`` (e.g. PyTorch CUDA tensors),
        which is wrapped in a zero-copy CuPy view via ``cupy.asarray``.
        """
        import cupy as cp

        if isinstance(tensor, cp.ndarray):
            return tensor
        if hasattr(tensor, "__cuda_array_interface__"):
            return cp.asarray(tensor)
        raise TypeError(
            f"Expected a CuPy ndarray or a CUDA tensor with "
            f"__cuda_array_interface__, got {type(tensor)}"
        )


# ---------------------------------------------------------------------------
# In-memory cache
# ---------------------------------------------------------------------------


class _BinaryCache:
    """Thread-safe in-memory cache of compiled binary kernels."""

    def __init__(self) -> None:
        self._store: Dict[str, TileIRKernel] = {}
        self._lock = threading.RLock()

    def _key(self, analysis: BinaryConfigAnalysis) -> str:
        """SHA-256 of the config JSON, truncated to 32 hex chars."""
        h = hashlib.sha256()
        h.update(analysis.config.to_json().encode())
        return h.hexdigest()[:32]

    def get_or_compile(self, analysis: BinaryConfigAnalysis) -> TileIRKernel:
        """Return cached kernel or build + cache it."""
        key = self._key(analysis)
        with self._lock:
            if key in self._store:
                return self._store[key]

            kernel = _compile_binary_from_scratch(analysis, key)
            self._store[key] = kernel
            return kernel


_binary_cache = _BinaryCache()


# ---------------------------------------------------------------------------
# Full compile pipeline
# ---------------------------------------------------------------------------

_KERNEL_PREFIX = "tileir_"


def _compile_binary_from_scratch(
    analysis: BinaryConfigAnalysis,
    cache_key: str,
) -> TileIRKernel:
    """Build IR → passes → bytecode → cubin → CuPy kernel for binary."""
    from etops.backends._tileir.ir_builder import build_binary_function

    kernel_name = _KERNEL_PREFIX + cache_key
    func_ir, ctx, temp_dir = build_binary_function(analysis, kernel_name)

    tokens = _dump_tokens()

    if tokens & {"tileir_before", "tileir_all"}:
        _dump_ir("BEFORE passes", func_ir)

    _apply_passes(func_ir, tokens)

    if tokens & {"tileir_after", "tileir_all"}:
        _dump_ir("AFTER all passes", func_ir)

    cubin_path = _build_cubin(func_ir, temp_dir, kernel_name)

    if "sass" in tokens:
        _dump_sass(str(cubin_path))

    return TileIRKernel(
        cubin_path=str(cubin_path),
        kernel_name=kernel_name,
        grid_size=max(analysis.grid_size, 1),
    )


def compile_binary_analysis(analysis: BinaryConfigAnalysis) -> TileIRKernel:
    """Compile a binary *analysis* to an executable kernel (cached).

    This is the main entry point used by :func:`tileir.create_operation`
    for binary contractions.

    Args:
        analysis: Frozen binary config analysis.

    Returns:
        A :class:`TileIRKernel` ready for launch.
    """
    return _binary_cache.get_or_compile(analysis)


# ---------------------------------------------------------------------------
# Unary compile pipeline
# ---------------------------------------------------------------------------


class _UnaryCache:
    """Thread-safe in-memory cache of compiled unary kernels."""

    def __init__(self) -> None:
        self._store: Dict[str, TileIRKernel] = {}
        self._lock = threading.RLock()

    def _key(self, analysis: UnaryConfigAnalysis) -> str:
        """SHA-256 of the config JSON, truncated to 32 hex chars."""
        h = hashlib.sha256()
        h.update(analysis.config.to_json().encode())
        return h.hexdigest()[:32]

    def get_or_compile(self, analysis: UnaryConfigAnalysis) -> TileIRKernel:
        """Return cached kernel or build + cache it."""
        key = self._key(analysis)
        with self._lock:
            if key in self._store:
                return self._store[key]

            kernel = _compile_unary_from_scratch(analysis, key)
            self._store[key] = kernel
            return kernel


_unary_cache = _UnaryCache()

_UNARY_KERNEL_PREFIX = "tileir_u_"


def _compile_unary_from_scratch(
    analysis: UnaryConfigAnalysis,
    cache_key: str,
) -> TileIRKernel:
    """Build IR → passes → bytecode → cubin → CuPy kernel for unary."""
    from etops.backends._tileir.ir_builder import build_unary_function

    kernel_name = _UNARY_KERNEL_PREFIX + cache_key
    func_ir, ctx, temp_dir = build_unary_function(analysis, kernel_name)

    tokens = _dump_tokens()

    if tokens & {"tileir_before", "tileir_all"}:
        _dump_ir("BEFORE passes (unary)", func_ir)

    _apply_passes(func_ir, tokens)

    if tokens & {"tileir_after", "tileir_all"}:
        _dump_ir("AFTER all passes (unary)", func_ir)

    cubin_path = _build_cubin(func_ir, temp_dir, kernel_name)

    if "sass" in tokens:
        _dump_sass(str(cubin_path))

    return TileIRKernel(
        cubin_path=str(cubin_path),
        kernel_name=kernel_name,
        grid_size=max(analysis.grid_size, 1),
        is_unary=True,
    )


def compile_unary_analysis(analysis: UnaryConfigAnalysis) -> TileIRKernel:
    """Compile a unary *analysis* to an executable kernel (cached).

    This is the main entry point used by :func:`tileir.create_operation`
    for unary operations.

    Args:
        analysis: Frozen unary config analysis.

    Returns:
        A :class:`TileIRKernel` ready for launch.
    """
    return _unary_cache.get_or_compile(analysis)
