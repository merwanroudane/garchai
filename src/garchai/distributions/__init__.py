"""Standardised conditional distributions for GARCH innovations.

All distributions here have zero mean and unit variance, so that the model's
``h_t`` is genuinely the conditional variance. See :mod:`garchai.distributions.base`.
"""

from .base import Distribution
from .ged import GED
from .normal import Normal
from .skew_t import SkewT
from .student_t import StudentT, StudentTKernel

__all__ = ["Distribution", "Normal", "StudentT", "StudentTKernel", "SkewT", "GED"]

_REGISTRY = {
    "normal": Normal,
    "gaussian": Normal,
    "student-t": StudentT,
    "studentt": StudentT,
    "t": StudentT,
    "student-t-kernel": StudentTKernel,
    "skew-t": SkewT,
    "skewt": SkewT,
    "ged": GED,
}


def get_distribution(name: str, **kwargs) -> Distribution:
    """Build a distribution by name.

    Examples
    --------
    >>> get_distribution("student-t", df=5.0).name
    'student-t'
    """
    key = name.lower().replace("_", "-")
    if key not in _REGISTRY:
        raise KeyError(
            f"Unknown distribution {name!r}. Available: {sorted(set(_REGISTRY))}"
        )
    return _REGISTRY[key](**kwargs)
