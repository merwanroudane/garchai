"""The differentiable variance cell: the primitive everything else rests on.

A :class:`VarianceCell` owns a GARCH-family recursion and nothing else. It holds
no data, no mean model and no scaling. It exposes one step of the recursion,
plus a ``filter`` that runs the whole path, and it keeps every intermediate
state inside the autograd graph.

Two rules are non-negotiable (blueprint sec. 16 and sec. 132):

* ``h_prev`` and ``eps_prev`` are **never** detached. Truncating the gradient is
  a deliberate algorithmic choice, requested explicitly through
  ``truncate_bptt``, never a silent default.
* Both raw and transformed parameters are inspectable, so a user can see the
  econometric quantity (``alpha = 0.08``) and not just the unconstrained real
  the optimiser moves.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn as nn
from torch import Tensor

from ..core.config import NumericalConfig

InitMethod = Literal[
    "arch_backcast",
    "tsgarch_backcast",
    "sample",
    "first_n",
    "mean_square",
    "unconditional",
    "trainable",
]

_INIT_DOC = """
Available ``init_method`` values, with the exact formula each uses.

``"arch_backcast"`` (default)
    The convention in :mod:`arch`. With ``tau = min(75, T)`` and weights
    ``w_i = 0.94^i`` normalised to sum to one over ``i < tau``,
    ``h_1 = sum_i w_i * eps_i^2``.

``"tsgarch_backcast"``
    The convention in the R package ``tsgarch``, with ``lambda = 0.7``:
    ``h_1 = lambda^T * mean(eps^2) + (1 - lambda) * sum_i lambda^i * eps_i^2``
    over the whole sample. The first term replaces the truncated tail with the
    sample mean square.

``"sample"``
    Full-sample variance of ``eps``.

``"first_n"``
    Mean square of the first ``sample_n`` observations (``tsgarch``'s option
    confusingly named ``"sample"``; default ``sample_n = 10``).

``"mean_square"``
    Full-sample mean square of ``eps`` (``tsgarch``'s option confusingly named
    ``"unconditional"``).

``"unconditional"``
    The *model-implied* long-run variance ``omega / (1 - persistence)``. Falls
    back to the sample variance when the process is not stationary.

``"trainable"``
    ``h_1`` is a learned parameter.

The two packages genuinely disagree, so ``garchai`` implements both rather than
splitting the difference. Choose ``"arch_backcast"`` when cross-checking against
:mod:`arch`, ``"tsgarch_backcast"`` when cross-checking against ``tsgarch``, and
whatever the paper specifies when reproducing a paper.
"""


@dataclass
class CellState:
    """State carried between two steps of a variance recursion.

    Attributes
    ----------
    h
        Conditional variance :math:`h_t`. Always present.
    log_h
        Log variance, for cells that recurse on the log scale (EGARCH). ``None``
        for cells that recurse on the level.
    extra
        Anything else a cell needs to carry: an RNN hidden state for RECH, a
        cell state for the embedded LSTM cells, a lag buffer for FIGARCH.
    """

    h: Tensor
    log_h: Tensor | None = None
    extra: dict[str, Tensor] | None = None

    def detach(self) -> CellState:
        """Return a detached copy. Used only for explicit truncated BPTT."""
        return CellState(
            h=self.h.detach(),
            log_h=None if self.log_h is None else self.log_h.detach(),
            extra=None if self.extra is None else {k: v.detach() for k, v in self.extra.items()},
        )


class VarianceCell(nn.Module, ABC):
    """Base class for a differentiable conditional-variance recursion.

    Subclasses implement :meth:`transformed_parameters` and :meth:`step`; the
    filter loop, initialisation and diagnostics are inherited.

    Parameters
    ----------
    init_method
        How :math:`h_1` is set. See :data:`_INIT_DOC` for the exact formula of
        each option. Paper reproductions must use whatever the paper specified —
        RECH, for instance, fixes :math:`\\sigma_1^2 = \\sigma_0^2` (Eq. 8b).
    backcast_lambda
        Decay factor. Ignored unless ``init_method`` is a backcast; each
        backcast convention supplies its own default (0.94 for ``arch``, 0.7 for
        ``tsgarch``) when this is left at ``None``.
    sample_n
        Number of leading observations used by ``init_method="first_n"``.
    numerical
        Shared numerical guards.
    """

    constraint_family: str = "generic"

    ARCH_LAMBDA = 0.94
    ARCH_TAU = 75
    TSGARCH_LAMBDA = 0.7

    def __init__(
        self,
        init_method: InitMethod = "arch_backcast",
        backcast_lambda: float | None = None,
        sample_n: int = 10,
        numerical: NumericalConfig | None = None,
    ):
        super().__init__()
        self.init_method = init_method
        self.backcast_lambda = backcast_lambda
        self.sample_n = sample_n
        self.numerical = numerical or NumericalConfig()
        if init_method == "trainable":
            self.raw_h0 = nn.Parameter(torch.zeros(()))
        else:
            self.register_parameter("raw_h0", None)

    # -- interface ---------------------------------------------------------- #
    @abstractmethod
    def transformed_parameters(self) -> dict[str, Tensor]:
        """Econometric parameters, after the constraint transform."""

    @abstractmethod
    def step(
        self, eps_prev: Tensor, state: CellState, params: dict[str, Tensor] | None = None
    ) -> tuple[Tensor, CellState]:
        """Advance one period.

        Parameters
        ----------
        eps_prev
            The innovation :math:`\\epsilon_{t-1}`.
        state
            State after period :math:`t-1`.
        params
            Pre-transformed parameters. Supplied by :meth:`filter` so the
            constraint transforms are evaluated once per pass instead of once
            per time step — that alone was over half the cost of filtering a
            4000-observation series. ``None`` means compute them here.

        Returns
        -------
        h_t, new_state
            The conditional variance for period ``t`` and the updated state.
        """

    def affine_terms(
        self, eps: Tensor, params: dict[str, Tensor]
    ) -> tuple[Tensor, Tensor] | None:
        r"""Express the recursion as :math:`h_t = c_{t-1} + \beta h_{t-1}`, if possible.

        Returns ``(c, beta)`` where ``c[j]`` is the term contributed by period
        ``j`` and ``beta`` is the scalar coefficient on the lagged variance, or
        ``None`` when the recursion is not affine in :math:`h_{t-1}`.

        GARCH and GJR are affine and get the fast vectorised path. EGARCH is not:
        its :math:`z_{t-1} = \epsilon_{t-1}/\sqrt{h_{t-1}}` term makes the
        recursion genuinely nonlinear in the lagged variance, so it falls back to
        the step loop.
        """
        return None

    @abstractmethod
    def persistence(self) -> Tensor:
        """Volatility persistence, on the scale natural to the family."""

    # -- shared machinery --------------------------------------------------- #
    def raw_parameters(self) -> dict[str, Tensor]:
        """The unconstrained reals the optimiser actually moves."""
        return {n: p.detach().clone() for n, p in self.named_parameters()}

    def unconditional_variance(self) -> Tensor:
        """Long-run variance :math:`\\omega/(1 - P)`, where ``P`` is persistence.

        Returns ``inf`` when the persistence is at or above one. Subclasses whose
        long-run variance has a different form override this.
        """
        params = self.transformed_parameters()
        p = self.persistence()
        denom = 1.0 - p
        return torch.where(
            denom > 0, params["omega"] / denom.clamp(min=1e-12), torch.full_like(denom, float("inf"))
        )

    def initial_variance(self, eps: Tensor) -> Tensor:
        """Compute :math:`h_1` according to ``init_method``. See :data:`_INIT_DOC`."""
        floor = self.numerical.variance_floor
        method = self.init_method

        if method == "trainable":
            from ..constraints.transforms import positive

            return positive(self.raw_h0, self.numerical.eps).clamp(min=floor)
        if method == "sample":
            return eps.var(unbiased=True).clamp(min=floor)
        if method == "mean_square":
            return eps.square().mean().clamp(min=floor)
        if method == "first_n":
            k = min(self.sample_n, eps.shape[-1])
            return eps[:k].square().mean().clamp(min=floor)
        if method == "unconditional":
            uv = self.unconditional_variance()
            if not torch.isfinite(uv):
                return eps.var(unbiased=True).clamp(min=floor)
            return uv.clamp(min=floor)
        if method == "arch_backcast":
            return self.arch_backcast(eps)
        if method == "tsgarch_backcast":
            return self.tsgarch_backcast(eps)
        raise ValueError(
            f"Unknown init_method {method!r}. Valid values: arch_backcast, "
            "tsgarch_backcast, sample, first_n, mean_square, unconditional, trainable."
        )

    def arch_backcast(self, eps: Tensor) -> Tensor:
        r"""Backcast exactly as :mod:`arch` computes it.

        .. math::
            h_1 = \sum_{i=0}^{\tau-1} w_i\, \epsilon_i^2, \qquad
            w_i = \frac{\lambda^i}{\sum_{j<\tau} \lambda^j}, \qquad
            \tau = \min(75, T),\; \lambda = 0.94

        Reproducing this formula bit for bit is what lets the numerical test
        suite assert agreement with ``arch`` to 1e-12 rather than merely
        "close after burn-in".
        """
        lam = self.ARCH_LAMBDA if self.backcast_lambda is None else self.backcast_lambda
        tau = min(self.ARCH_TAU, eps.shape[-1])
        power = torch.arange(tau, dtype=eps.dtype, device=eps.device)
        weights = lam**power
        weights = weights / weights.sum()
        return (weights * eps[:tau].square()).sum().clamp(min=self.numerical.variance_floor)

    def tsgarch_backcast(self, eps: Tensor) -> Tensor:
        r"""Backcast exactly as the R package ``tsgarch`` computes it.

        .. math::
            h_1 = \lambda^{T}\,\overline{\epsilon^2}
                  + (1-\lambda)\sum_{i=0}^{T-1}\lambda^{i}\epsilon_i^2,
            \qquad \lambda = 0.7

        The leading term stands in for the truncated tail of the geometric sum,
        so that :math:`\lambda \to 1` recovers the sample mean square.
        """
        lam = self.TSGARCH_LAMBDA if self.backcast_lambda is None else self.backcast_lambda
        n = eps.shape[-1]
        sq = eps.square()
        power = torch.arange(n, dtype=eps.dtype, device=eps.device)
        weighted = ((lam**power) * sq).sum()
        value = (lam**n) * sq.mean() + (1.0 - lam) * weighted
        return value.clamp(min=self.numerical.variance_floor)

    def h1_from_backcast(self, backcast: Tensor) -> Tensor:
        r"""Turn a pre-sample backcast into :math:`h_1`.

        The two packages differ here as well, and the difference is conceptual,
        not cosmetic:

        * ``tsgarch`` uses the backcast **as** :math:`h_1`.
        * ``arch`` treats the backcast as the pre-sample state standing in for
          *both* :math:`\epsilon_0^2` and :math:`h_0`, then runs one step of the
          recursion. For GARCH(1,1) that gives
          :math:`h_1 = \omega + (\alpha+\beta)\,\mathrm{bc}`.

        The base implementation returns the backcast unchanged (``tsgarch``
        behaviour). Cells override it to supply their family's one-step form,
        which is what ``init_method="arch_backcast"`` then uses.
        """
        return backcast

    def initial_state(self, eps: Tensor) -> CellState:
        """Build the state for period 1."""
        if self.init_method == "arch_backcast":
            h1 = self.h1_from_backcast(self.arch_backcast(eps))
        else:
            h1 = self.initial_variance(eps)
        h1 = h1.clamp(min=self.numerical.variance_floor)
        return CellState(h=h1, log_h=torch.log(h1))

    def filter(
        self,
        eps: Tensor,
        state: CellState | None = None,
        truncate_bptt: int | None = None,
    ) -> tuple[Tensor, CellState]:
        r"""Run the recursion over a whole innovation series.

        Parameters
        ----------
        eps
            Innovations :math:`\epsilon_1, \dots, \epsilon_T`, shape ``(T,)``.
        state
            Optional starting state, for continuing a filter across chunks. When
            ``None``, :meth:`initial_state` is used.
        truncate_bptt
            If given, detach the state every ``k`` steps. This is truncated
            backpropagation through time and it **changes the gradient**; it is
            never applied unless asked for.

        Returns
        -------
        h, final_state
            ``h`` has shape ``(T,)`` and is aligned with ``eps``: ``h[t]`` is the
            variance of ``eps[t]``, conditional on information through ``t-1``.
        """
        if eps.ndim != 1:
            raise ValueError(f"filter expects a 1-D innovation series, got shape {tuple(eps.shape)}.")
        if state is None:
            state = self.initial_state(eps)

        params = self.transformed_parameters()

        # Fast path: affine recursions have a closed form, so we can avoid the
        # Python loop entirely. Truncated BPTT deliberately opts out, since the
        # whole point of it is to cut the graph at fixed intervals.
        if truncate_bptt is None:
            terms = self.affine_terms(eps, params)
            if terms is not None:
                return self._filter_affine(eps, state, *terms)

        n = eps.shape[0]
        out = [state.h]
        for t in range(1, n):
            h_t, state = self.step(eps[t - 1], state, params)
            if truncate_bptt and t % truncate_bptt == 0:
                state = state.detach()
            out.append(h_t)
        return torch.stack(out), state

    def _filter_affine(
        self, eps: Tensor, state: CellState, c: Tensor, beta: Tensor
    ) -> tuple[Tensor, CellState]:
        r"""Closed-form evaluation of :math:`h_t = c_{t-1} + \beta h_{t-1}`.

        Unrolling the recursion from a chunk start :math:`t_0` gives

        .. math::
            h_{t_0+i} = \beta^{i} h_{t_0}
                      + \beta^{\,i-1}\sum_{m=0}^{i-1}\beta^{-m} c_{t_0+m},

        so the inner sum is a single ``cumsum`` and the whole chunk is computed
        without iterating.

        The catch is :math:`\beta^{-m}`, which overflows for large ``m``. We
        therefore process in chunks whose length is chosen from ``beta`` so that
        :math:`\beta^{-(K-1)}` stays below 1e150 — comfortably inside float64.
        For a typical estimated ``beta`` near 0.9 that permits chunks of a
        thousand-plus observations, reducing a 4000-step Python loop to a
        handful of vectorised passes.

        No per-step variance floor is applied on this path, and none is needed:
        an affine cell has ``omega > 0`` and non-negative coefficients acting on
        non-negative inputs, so :math:`h_t \ge \omega` throughout and the floor
        can never bind.
        """
        n = eps.shape[0]
        dtype, device = eps.dtype, eps.device
        h1 = state.h
        if n == 1:
            return h1.reshape(1), CellState(h=h1)

        beta_value = float(beta.detach())
        if beta_value <= 0.0:
            out = torch.cat([h1.reshape(1), c[: n - 1]])
            return out, CellState(h=out[-1])

        # Largest chunk for which beta**-(K-1) stays representable.
        max_exponent = 150.0 * math.log(10.0)
        span = max_exponent / max(-math.log(beta_value), 1e-12)
        chunk = int(max(1, min(n, math.floor(span))))

        pieces: list[Tensor] = [h1.reshape(1)]
        current = h1
        index = 0
        while index < n - 1:
            size = min(chunk, n - 1 - index)
            block = c[index : index + size]
            i = torch.arange(1, size + 1, dtype=dtype, device=device)
            m = torch.arange(size, dtype=dtype, device=device)

            scaled = torch.pow(beta, -m) * block
            cumulative = torch.cumsum(scaled, dim=0)
            values = torch.pow(beta, i) * current + torch.pow(beta, i - 1.0) * cumulative

            pieces.append(values)
            current = values[-1]
            index += size

        out = torch.cat(pieces)
        return out, CellState(h=out[-1])

    def summary_row(self) -> dict[str, float]:
        """Flat dict of the transformed parameters plus persistence, for tables."""
        row = {k: float(v) for k, v in self.transformed_parameters().items() if v.numel() == 1}
        row["persistence"] = float(self.persistence())
        uv = self.unconditional_variance()
        row["long_run_variance"] = float(uv)
        return row

    def extra_repr(self) -> str:  # pragma: no cover - cosmetic
        params = ", ".join(
            f"{k}={float(v):.4f}"
            for k, v in self.transformed_parameters().items()
            if v.numel() == 1
        )
        return f"init={self.init_method}, {params}"
