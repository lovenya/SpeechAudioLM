from __future__ import annotations


class EvaluationError(RuntimeError):
    """Raised for user-actionable evaluation setup errors."""


class MetricUnavailableError(EvaluationError):
    """Raised when a selected metric is missing its backend or model assets."""

