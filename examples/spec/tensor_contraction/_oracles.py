"""Independent NumPy references for the tensor-contraction spec artifacts.

Each entry calls ``np.einsum`` with the contraction's exact einsum string
rather than inferring it from the parsed IR, so a stride or schedule typo
in any ``.teir`` file produces a visible test failure instead of agreeing
with a buggy IR-derived oracle. Keys match the artifact name returned by
``_build_artifacts`` — ``<einsum_family>/<schedule>`` for files that live
under a per-family subdirectory.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

__all__ = ["REFERENCES"]


REFERENCES: dict[str, Callable[..., np.ndarray]] = {
    "trus_pqtu_pqrs/scalar": lambda in0, in1: np.einsum(
        "trus,pqtu->pqrs", in0, in1, optimize=False
    ),
    "trus_pqtu_pqrs/gemm": lambda in0, in1: np.einsum(
        "trus,pqtu->pqrs", in0, in1, optimize=False
    ),
    "trus_pqtu_pqrs/brgemm": lambda in0, in1: np.einsum(
        "trus,pqtu->pqrs", in0, in1, optimize=False
    ),
    "abcd_efab_efcd/brgemm": lambda in0, in1: np.einsum(
        "abcd,efab->efcd", in0, in1, optimize=False
    ),
    "acbd_eafb_ecfd/brgemm": lambda in0, in1: np.einsum(
        "acbd,eafb->ecfd", in0, in1, optimize=False
    ),
    "dba_dac_dbc/scalar": lambda in0, in1: np.einsum(
        "dba,dac->dbc", in0, in1, optimize=False
    ),
    "dba_dac_dbc/scalar_reordered": lambda in0, in1: np.einsum(
        "dba,dac->dbc", in0, in1, optimize=False
    ),
    # _cases.py entries.
    "yxgcaei_yxhfca_yhgfxei/scalar": lambda in0, in1: np.einsum(
        "yxgcaei,yxhfca->yhgfxei", in0, in1, optimize=False
    ),
}
