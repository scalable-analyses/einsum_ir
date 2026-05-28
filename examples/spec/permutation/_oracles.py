"""Independent NumPy references for the permutation spec artifacts.

Each entry re-encodes the permutation directly via ``np.transpose`` rather
than inferring it from the parsed IR, so a stride typo in any ``.teir``
file produces a visible test failure instead of agreeing with a buggy
IR-derived oracle. Keys match the artifact name returned by
``_build_artifacts`` — ``<einsum_family>/<descriptor>`` for both ``.teir``
files (under a per-family subdirectory) and ``_cases.py`` entries.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

__all__ = ["REFERENCES"]


REFERENCES: dict[str, Callable[..., np.ndarray]] = {
    # .teir files, grouped by einsum family.
    "abcd_dcba/scalar": lambda in0: np.transpose(in0, (3, 2, 1, 0)),
    "abcd_dcba/tiled": lambda in0: np.transpose(in0, (3, 2, 1, 0)),
    "behi_ehib/scalar": lambda in0: np.transpose(in0, (1, 2, 3, 0)),
    "acdb_abcd/scalar": lambda in0: np.transpose(in0, (0, 3, 1, 2)),
    # _cases.py entries, grouped by einsum family (slashes match the dict keys).
    "a_a/scalar_f64": np.copy,
    "abc_cba/scalar_f64": lambda in0: np.transpose(in0, (2, 1, 0)),
    "rank9_arbitrary/scalar": lambda in0: np.transpose(
        in0, (2, 1, 4, 0, 5, 7, 3, 8, 6)
    ),
    "rank9_same_inner/scalar": lambda in0: np.transpose(
        in0, (2, 1, 4, 0, 5, 7, 3, 6, 8)
    ),
    "abc_bca/strided_input": lambda in0: np.transpose(in0, (1, 2, 0)),
}
