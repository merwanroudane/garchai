"""Forecast evaluation: metrics and formal comparison tests."""

from .comparison import (
    ComparisonResult, diebold_mariano, giacomini_white,
    model_confidence_set, newey_west_variance,
)

__all__ = [
    "ComparisonResult", "newey_west_variance", "diebold_mariano",
    "giacomini_white", "model_confidence_set",
]
