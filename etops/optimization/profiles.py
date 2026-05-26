"""Hardware-derived target descriptors and backend strategies.

The profile is the product of four orthogonal dimensions:

* `Microarchitecture` — per-design cache geometry. Hardware-only, ISA-
  independent.
* `ISAExtension` — vector/matrix unit capability plus egister-tight kernel
   layout (m-factor x n-factor of accumulator registers).
* `BackendStrategy` — backend-specific tuning policy: tile-shape regime
  (`register` for libxsmm-style JIT, `library` for cblas-style external
  kernels), role-cardinality targets, and the cache / parallelism
  heuristics.
* dtype — late-bound; passed to `OptimizationProfile.tile_targets`
  per primitive.

The default factories (`tpp_profile`, `blas_profile`) **fully runtime-
discover** the microarchitecture, ISA extension, and core count from
`/sys`, `/proc/cpuinfo`, and `sysctl`. The baked tables in this module are
documented offline defaults.

Per-ISA register-blocking values are sourced from libxsmm's GEMM kernel
generators (see references on each entry).
"""

from __future__ import annotations

import logging
import os
import platform
import subprocess
from dataclasses import dataclass
from pathlib import Path

from etops.ir.dtypes import DataType

__all__ = [
    "ISA_EXTENSIONS",
    "BackendStrategy",
    "BlasStrategy",
    "EffectiveTileTargets",
    "ISAExtension",
    "Microarchitecture",
    "OptimizationProfile",
    "TppStrategy",
    "blas_profile",
    "detect_isa_extension",
    "detect_microarch",
    "detect_num_threads",
    "tpp_profile",
]

_LOG = logging.getLogger(__name__)


# --------------------------------------------------------------------------
# Dataclasses
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Microarchitecture:
    """Per-design cache geometry. ISA-independent.

    `l2_bytes` is the per-core L2 effective capacity. On shared-L2
    designs callers are responsible for dividing by active sibling
    count when targeting multi-threaded kernels.

    `l3_bytes` is the per-core LLC slice for inclusive / non-inclusive
    caches and the shared LLC for chiplet designs. Treat as advisory;
    the runtime detection path reads exact values from /sys when available.
    """

    name: str
    l1_bytes: int
    l2_bytes: int
    l3_bytes: int


@dataclass(frozen=True)
class ISAExtension:
    """ISA capabilities + libxsmm-tested register-tight kernel layout.

    For SIMD-only ISAs (`matrix_tile_bytes == 0`):
        ``M_elements(dtype) = m_factor * (vector_bytes // dtype.bytes)``
        ``N_elements        = n_factor``  (raw register-column count)

    For matrix-unit ISAs (`matrix_tile_bytes > 0`, e.g. SME):
        ``M_elements = N_elements = m_factor * (matrix_tile_bytes // dtype.bytes)``
        (`n_factor` is unused; the kernel is symmetric)
    """

    name: str
    vector_bytes: int
    matrix_tile_bytes: int
    m_factor: int
    n_factor: int


@dataclass(frozen=True)
class BackendStrategy:
    """Backend-specific tuning policy. Hardware-independent.

    `tile_policy`:
        * ``"register"`` — TEIR tile size IS the libxsmm-style register-
          blocked kernel size; use the extension's `m_factor` / `n_factor`.
        * ``"library"`` — TEIR hands the tile as a block to an external
          library (cblas) that tiles internally; use a flat
          `library_dim_elements` for M and N.

    `k_over_mn` sets the K depth as a multiple of ``max(M, N)``. The
    default 4 is well above the C-accumulator amortization threshold
    and stays inside L1 for narrow-N register kernels.
    """

    name: str
    role_target_m: int
    role_target_n: int
    role_target_k: int
    tile_policy: str
    library_dim_elements: int
    k_over_mn: int
    parallel_l2_fraction: float
    cache_block_l3_fraction: float
    divisor_slack: float
    parallel_min_fanout: int
    compensation_min: float
    compensation_max: float


@dataclass(frozen=True)
class EffectiveTileTargets:
    """Resolved per-dtype tile-size targets handed to tiling passes."""

    m: int
    n: int
    k: int
    batch_reduce: int


@dataclass(frozen=True)
class OptimizationProfile:
    """Composition of microarch, ISA extension, backend strategy, and core count."""

    microarch: Microarchitecture
    extension: ISAExtension
    backend: BackendStrategy
    num_threads: int

    def tile_targets(self, dtype: DataType) -> EffectiveTileTargets:
        """Resolve tile-size targets for ``dtype``."""

        b = self.backend
        e = self.extension
        if b.tile_policy == "library":
            m = n = b.library_dim_elements
        elif e.matrix_tile_bytes:
            matrix_lanes = max(e.matrix_tile_bytes // dtype.bytes, 1)
            m = n = e.m_factor * matrix_lanes
        else:
            vector_lanes = max(e.vector_bytes // dtype.bytes, 1)
            m = e.m_factor * vector_lanes
            n = e.n_factor
        k = b.k_over_mn * max(m, n)
        return EffectiveTileTargets(m=m, n=n, k=k, batch_reduce=b.role_target_k)


# --------------------------------------------------------------------------
# ISA register-blocking catalog
# --------------------------------------------------------------------------

#: Per-ISA-extension register-blocking layout, sourced from libxsmm's
#: GEMM kernel generators (paths under ``src/`` upstream):
#:
#:   * ``neon``  — generator_gemm_common_aarch64.c:2185 (V8.2+ branch,
#:                 ``4N+5<=32``). The V8.1-only branch at :2181 uses
#:                 ``5N+4<=32`` (N=5); every modern AArch64 target
#:                 (Apple M1/M4, Neoverse V1/V2, SVE) takes the V8.2+
#:                 branch with N=6.
#:   * ``avx2``  — generator_gemm_sse_avx_avx2_avx512.c:1653 (max_n=3).
#:   * ``avx512``— generator_gemm_sse_avx_avx2_avx512.c:399 (``4N+5<=32``).
#:   * ``sme``   — generator_gemm_sme.c:30 (default 32x32 = 2x2 ZA grid).
ISA_EXTENSIONS: dict[str, ISAExtension] = {
    "neon": ISAExtension("neon", 16, 0, 4, 6),
    "avx2": ISAExtension("avx2", 32, 0, 4, 3),
    "avx512": ISAExtension("avx512", 64, 0, 4, 6),
    "sme": ISAExtension("sme", 16, 64, 2, 2),
}


#: Conservative microarch used when runtime cache probing fails.
_FALLBACK_MICROARCH = Microarchitecture(
    name="fallback",
    l1_bytes=32 * 1024,
    l2_bytes=256 * 1024,
    l3_bytes=2 * 1024 * 1024,
)


# --------------------------------------------------------------------------
# Detection
# --------------------------------------------------------------------------


def detect_num_threads() -> int:
    """Return the number of CPUs available to this process."""

    try:
        affinity = os.sched_getaffinity(0)
    except (AttributeError, OSError):
        return os.cpu_count() or 1
    return len(affinity) or 1


def _read_int(path: Path) -> int | None:
    try:
        text = path.read_text().strip()
    except OSError:
        return None
    if not text:
        return None
    multiplier = 1
    if text[-1] in "KMG":
        multiplier = {"K": 1024, "M": 1024**2, "G": 1024**3}[text[-1]]
        text = text[:-1]
    try:
        return int(text) * multiplier
    except ValueError:
        return None


def _sysctl(key: str) -> str | None:
    try:
        result = subprocess.run(
            ["sysctl", "-n", key],
            capture_output=True,
            text=True,
            timeout=1,
            check=False,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip() or None


def _sysctl_int(key: str) -> int | None:
    raw = _sysctl(key)
    if raw is None:
        return None
    try:
        return int(raw)
    except ValueError:
        return None


def _cache_topology_linux() -> tuple[int, int, int] | None:
    """Return ``(l1d, l2, l3)`` for cpu0 from /sys, or None if unavailable.

    Reads index1 (L1d), index2 (L2 unified), index3 (L3 / LLC). L3 may
    be absent on systems without a shared LLC; the caller treats a 0
    here as "fill in the fallback".
    """

    base = Path("/sys/devices/system/cpu/cpu0/cache")
    if not base.exists():
        return None
    sizes = []
    for idx, _name in ((1, "L1d"), (2, "L2"), (3, "L3")):
        sz = _read_int(base / f"index{idx}" / "size")
        sizes.append(sz)
    if sizes[0] is None or sizes[1] is None:
        return None
    return sizes[0], sizes[1], sizes[2] if sizes[2] is not None else 0


def _cache_topology_darwin() -> tuple[int, int, int] | None:
    """Return ``(l1d, l2, l3)`` from Darwin sysctls, or None if unavailable.

    Prefers ``hw.perflevel0.*`` (Apple Silicon big.LITTLE) when present.
    """

    def _level(suffix: str) -> int | None:
        return _sysctl_int(f"hw.perflevel0.{suffix}") or _sysctl_int(f"hw.{suffix}")

    l1d = _level("l1dcachesize")
    l2 = _level("l2cachesize")
    l3 = _level("l3cachesize") or 0
    if l1d is None or l2 is None:
        return None
    return l1d, l2, l3


def detect_microarch() -> Microarchitecture:
    """Probe the host's cache topology at runtime.

    Linux reads ``/sys/devices/system/cpu/cpu0/cache/index*``. Darwin
    reads ``hw.perflevel0.*`` (Apple Silicon performance-cluster
    values), with ``hw.*`` as a secondary path. Falls back to
    `_FALLBACK_MICROARCH` when no probe succeeds.
    """

    if platform.system() == "Linux":
        probed = _cache_topology_linux()
    elif platform.system() == "Darwin":
        probed = _cache_topology_darwin()
    else:
        probed = None
    if probed is None:
        return _FALLBACK_MICROARCH
    l1d, l2, l3 = probed
    return Microarchitecture(
        name="host",
        l1_bytes=l1d,
        l2_bytes=l2,
        l3_bytes=l3 or _FALLBACK_MICROARCH.l3_bytes,
    )


def _cpuinfo_value(cpuinfo: str, key: str) -> str | None:
    for line in cpuinfo.splitlines():
        if line.startswith(key):
            _, _, value = line.partition(":")
            return value.strip()
    return None


def detect_isa_extension() -> ISAExtension:
    """Return the widest available ISA extension on this machine.

    Linux: parses ``/proc/cpuinfo`` flags. Darwin arm64: probes the SME
    sysctl. The fallback is ``avx2`` on x86_64 and ``neon`` on aarch64.
    """

    system = platform.system()
    machine = platform.machine().lower()
    if system == "Darwin" and machine in ("arm64", "aarch64"):
        if _sysctl_int("hw.optional.arm.FEAT_SME") == 1:
            return ISA_EXTENSIONS["sme"]
        return ISA_EXTENSIONS["neon"]
    if system == "Linux":
        try:
            cpuinfo = Path("/proc/cpuinfo").read_text()
        except OSError:
            cpuinfo = ""
        flags = (
            _cpuinfo_value(cpuinfo, "flags")
            or _cpuinfo_value(cpuinfo, "Features")
            or ""
        )
        tokens = set(flags.split())
        if machine in ("aarch64", "arm64"):
            if "sme" in tokens:
                return ISA_EXTENSIONS["sme"]
            return ISA_EXTENSIONS["neon"]
        if "avx512f" in tokens:
            return ISA_EXTENSIONS["avx512"]
        if "avx2" in tokens:
            return ISA_EXTENSIONS["avx2"]
    if machine in ("aarch64", "arm64"):
        return ISA_EXTENSIONS["neon"]
    return ISA_EXTENSIONS["avx2"]


# --------------------------------------------------------------------------
# Strategy and profile factories
# --------------------------------------------------------------------------


def TppStrategy() -> BackendStrategy:
    """Default tuning for the libxsmm-backed TPP backend (register-blocked)."""

    return BackendStrategy(
        name="tpp",
        role_target_m=1,
        role_target_n=1,
        role_target_k=2,
        tile_policy="register",
        library_dim_elements=0,
        k_over_mn=4,
        parallel_l2_fraction=0.5,
        cache_block_l3_fraction=0.5,
        divisor_slack=1.5,
        parallel_min_fanout=4,
        compensation_min=0.5,
        compensation_max=4.0,
    )


def BlasStrategy() -> BackendStrategy:
    """Default tuning for the cblas-backed BLAS backend (library-blocked).

    BLAS does its own internal register-level tiling, so TEIR-level tiles
    are flat 512x512x512 blocks. The K-role target is 1 because cblas
    has no BRGEMM entry point; multi-K contractions stay outside the
    primitive.
    """

    return BackendStrategy(
        name="blas",
        role_target_m=1,
        role_target_n=1,
        role_target_k=1,
        tile_policy="library",
        library_dim_elements=512,
        k_over_mn=1,
        parallel_l2_fraction=0.5,
        cache_block_l3_fraction=0.5,
        divisor_slack=1.5,
        parallel_min_fanout=4,
        compensation_min=0.5,
        compensation_max=4.0,
    )


def tpp_profile() -> OptimizationProfile:
    """Auto-detected `OptimizationProfile` for the TPP backend."""

    return OptimizationProfile(
        microarch=detect_microarch(),
        extension=detect_isa_extension(),
        backend=TppStrategy(),
        num_threads=detect_num_threads(),
    )


def blas_profile() -> OptimizationProfile:
    """Auto-detected `OptimizationProfile` for the BLAS backend."""

    return OptimizationProfile(
        microarch=detect_microarch(),
        extension=detect_isa_extension(),
        backend=BlasStrategy(),
        num_threads=detect_num_threads(),
    )
