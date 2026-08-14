"""Central configuration objects.

Numerical settings live in one place so that every cell, likelihood and
forecaster shares the same floors and ceilings. Silent per-module constants are
how variance filters end up disagreeing with each other.
"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass, field, replace
from typing import Literal

import numpy as np
import torch

DType = Literal["float32", "float64"]

_TORCH_DTYPE = {"float32": torch.float32, "float64": torch.float64}


@dataclass(frozen=True)
class NumericalConfig:
    """Numerical guards shared by every variance recursion.

    Parameters
    ----------
    variance_floor
        Lower bound applied to :math:`h_t`. Prevents ``log(0)`` and division by
        zero in the likelihood when a model drives the variance toward zero.
    max_log_variance
        Ceiling on :math:`\\log h_t` for log-variance models (EGARCH, and any
        cell that exponentiates). ``exp(50)`` is already ~5e21; beyond that the
        float64 likelihood is meaningless.
    eps
        Small constant added inside ``softplus``-style positivity transforms.
    grad_clip
        Global gradient-norm clip. ``None`` disables clipping.
    stationarity_slack
        The ``delta`` in ``alpha + beta <= 1 - delta``. Keeps the persistence
        strictly below one so that the unconditional variance stays finite.
    min_df
        Floor on Student-t degrees of freedom. The variance of a standardised
        t exists only for ``nu > 2``; we keep a margin.
    """

    variance_floor: float = 1e-12
    max_log_variance: float = 50.0
    eps: float = 1e-8
    grad_clip: float | None = 5.0
    stationarity_slack: float = 1e-4
    min_df: float = 2.05

    def with_(self, **kwargs) -> NumericalConfig:
        """Return a copy with the given fields replaced."""
        return replace(self, **kwargs)


@dataclass(frozen=True)
class TrainConfig:
    """Optimisation settings for gradient-trained models.

    The defaults follow the protocol shared by Zhao et al. (2024) and Wei et al.
    (2025): Adam, early stopping with patience 20 on validation loss, and a
    scheduler that halves the learning rate when validation performance
    stagnates.
    """

    optimizer: Literal["adam", "adamw", "lbfgs"] = "adam"
    lr: float = 1e-2
    weight_decay: float = 0.0
    max_epochs: int = 500
    patience: int = 20
    lr_patience: int = 10
    lr_factor: float = 0.5
    min_lr: float = 1e-6
    validation_split: float = 0.0
    shuffle: bool = False  # never shuffle time points; see blueprint sec. 26
    verbose: bool = False
    refine_with_lbfgs: bool = False
    lbfgs_max_iter: int = 200

    def __post_init__(self) -> None:
        if self.shuffle:
            raise ValueError(
                "shuffle=True is not permitted for recurrent volatility models: "
                "shuffling individual time points destroys the recursion. Use "
                "contiguous windows or truncated BPTT instead."
            )
        if not 0.0 <= self.validation_split < 1.0:
            raise ValueError("validation_split must lie in [0, 1).")


@dataclass(frozen=True)
class ModelConfig:
    """Everything a model needs that is not a learnable parameter."""

    mean: Literal["zero", "constant", "ar", "arma", "neural"] = "constant"
    mean_order: tuple[int, int] = (0, 0)
    distribution: Literal["normal", "student-t", "skew-t", "ged"] = "normal"
    dtype: DType = "float64"
    device: str = "cpu"
    scale: float | Literal["auto"] | None = None
    seed: int | None = None
    enforce_stationarity: bool = True
    numerical: NumericalConfig = field(default_factory=NumericalConfig)

    @property
    def torch_dtype(self) -> torch.dtype:
        return _TORCH_DTYPE[self.dtype]

    def with_(self, **kwargs) -> ModelConfig:
        return replace(self, **kwargs)


def set_seed(seed: int, deterministic: bool = True) -> None:
    """Seed Python, NumPy and PyTorch (CPU and CUDA) in one call.

    Parameters
    ----------
    seed
        The seed value.
    deterministic
        When ``True``, also request deterministic cuDNN kernels. This costs
        speed but is what reproduction runs need.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():  # pragma: no cover - no GPU in CI
        torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False


def resolve_scale(y: np.ndarray | torch.Tensor, scale: float | str | None) -> float:
    """Resolve the ``scale`` option into a concrete multiplier.

    Financial daily returns are ~1e-2, which pushes ``omega`` to ~1e-6 and makes
    the likelihood surface badly conditioned. Most of the literature multiplies
    returns by 100. ``"auto"`` picks 100 when the sample standard deviation is
    below 1, and 1 otherwise.
    """
    if scale is None:
        return 1.0
    if isinstance(scale, (int, float)):
        if scale <= 0:
            raise ValueError("scale must be positive.")
        return float(scale)
    if scale == "auto":
        sd = float(torch.as_tensor(y, dtype=torch.float64).std())
        return 100.0 if sd < 1.0 else 1.0
    raise ValueError(f"Unrecognised scale: {scale!r}. Use None, a number, or 'auto'.")
