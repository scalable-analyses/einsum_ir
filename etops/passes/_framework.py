"""Pass framework: pass function signature, `PassContext`, `PassPipeline`."""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable
from dataclasses import dataclass

from etops.analyses import AnalysisManager
from etops.ir import Teir
from etops.optimization import OptimizationProfile

__all__ = ["Pass", "PassContext", "PassPipeline"]

_LOG = logging.getLogger(__name__)


@dataclass
class PassContext:
    """Per-run context threaded through every pass."""

    analyses: AnalysisManager
    profile: OptimizationProfile


#: A pass is a callable ``(teir, ctx) -> Teir``. Module-level functions
#: are the canonical form; the pipeline invalidates analyses whenever a
#: pass returns a different `Teir`.
Pass = Callable[[Teir, PassContext], Teir]


class PassPipeline:
    """Ordered sequence of pass callables."""

    def __init__(self, passes: Iterable[Pass]) -> None:
        self._passes: tuple[Pass, ...] = tuple(passes)

    @property
    def passes(self) -> tuple[Pass, ...]:
        return self._passes

    def run(
        self,
        teir: Teir,
        *,
        profile: OptimizationProfile,
    ) -> Teir:
        """Execute the pipeline against ``teir`` and return the result."""

        ctx = PassContext(analyses=AnalysisManager(), profile=profile)
        current = teir
        for p in self._passes:
            _LOG.debug("running pass %s", getattr(p, "__name__", repr(p)))
            new = p(current, ctx)
            if new is not current:
                ctx.analyses.invalidate_all_for(current)
                current = new
        return current
