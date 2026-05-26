"""Diagnostic exception types."""

from __future__ import annotations

__all__ = [
    "TeirEmissionError",
    "TeirError",
    "TeirLoweringError",
    "TeirPassError",
    "TeirRuntimeError",
    "TeirValidationError",
]


class TeirError(Exception):
    """Base class for every exception raised by the etops package.

    Attributes:
        message: Human-readable message.
        suggestion: Optional remediation hint.
    """

    def __init__(self, message: str, suggestion: str | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.suggestion = suggestion

    def __str__(self) -> str:
        if self.suggestion:
            return f"{self.message}\n  hint: {self.suggestion}"
        return self.message


class TeirValidationError(TeirError):
    """Raised when a `Teir` fails validation.

    Carries a `diagnostics` list of every collected violation; the first
    item duplicates the parent's ``message``.
    """

    def __init__(
        self,
        message: str,
        suggestion: str | None = None,
        diagnostics: list[TeirValidationError] | None = None,
    ) -> None:
        super().__init__(message, suggestion=suggestion)
        self.diagnostics: list[TeirValidationError] = list(diagnostics or [])

    def __str__(self) -> str:
        if not self.diagnostics or self.diagnostics == [self]:
            return super().__str__()
        head = f"{len(self.diagnostics)} validation error(s):"
        items = [
            f"  [{i + 1}] {entry.message}" for i, entry in enumerate(self.diagnostics)
        ]
        return "\n".join([head, *items])


class TeirEmissionError(TeirError):
    """Raised when an emitter (einsum, textual IR) cannot produce a `Teir`."""


class TeirPassError(TeirError):
    """Raised when a pass fails."""


class TeirLoweringError(TeirError):
    """Raised when a backend cannot lower a `Teir`."""


class TeirRuntimeError(TeirError):
    """Raised when execution fails."""
