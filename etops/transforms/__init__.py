"""Pure-function transformations on a `Teir`.

Every transformation takes an immutable `Teir` plus arguments and returns
a new immutable `Teir`. Transformations preserve semantic equivalence
within their documented preconditions and refuse otherwise.
"""

from __future__ import annotations

from etops.transforms._canonicalize import canonicalize_ids
from etops.transforms._iteration import fuse_iterations, set_policy, split_iteration

__all__ = [
    "canonicalize_ids",
    "fuse_iterations",
    "set_policy",
    "split_iteration",
]
