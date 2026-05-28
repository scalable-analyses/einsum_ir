"""Programmatic spec entries for tensor-contraction regressions.

Dict keys follow the same ``<einsum_family>/<descriptor>`` convention as
the textual ``.teir`` artifacts in this directory; the IR ``@name`` is
derived mechanically by replacing the slash with an underscore so the
identifier is a single token. Each :class:`Case` carries a callable that
returns a freshly-constructed :class:`~etops.ir.Teir`.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass

from etops.emit import einsum
from etops.ir import Teir

__all__ = ["CASES", "Case"]


@dataclass(frozen=True)
class Case:
    """One spec entry: a (re-)derivable IR plus its declared shape.

    ``skip_backends`` enumerates backend names for which the case is known
    not to lower. The spec test honors this list and reports a skip with
    a clear reason.
    """

    expr: str
    extents: Mapping[str, int]
    dtype: str
    build: Callable[[], Teir]
    skip_backends: tuple[str, ...] = ()
    skip_reason: str = ""


def _ir_name(key: str) -> str:
    """Mechanical IR ``@name`` derived from the artifact key.

    The artifact key uses ``<family>/<descriptor>``; the IR identifier
    must match ``[A-Za-z_][A-Za-z0-9_.\\-]*`` (no slash), so we collapse
    the separator into an underscore.
    """

    return key.replace("/", "_")


def _einsum_case(key: str, expr: str, extents: Mapping[str, int], dtype: str) -> Case:
    return Case(
        expr=expr,
        extents=extents,
        dtype=dtype,
        build=lambda: einsum(expr, dim_sizes=extents, dtype=dtype, name=_ir_name(key)),
    )


CASES: dict[str, Case] = {
    "yxgcaei_yxhfca_yhgfxei/scalar": _einsum_case(
        "yxgcaei_yxhfca_yhgfxei/scalar",
        "yxgcaei,yxhfca->yhgfxei",
        extents={
            "y": 4,
            "x": 3,
            "g": 6,
            "c": 7,
            "a": 2,
            "e": 8,
            "i": 3,
            "h": 4,
            "f": 5,
        },
        dtype="f32",
    ),
}
