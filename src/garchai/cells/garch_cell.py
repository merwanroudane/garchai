"""GARCH(1,1), GJR-GARCH(1,1,1) and EGARCH(1,1,1) as differentiable cells."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch import Tensor

from ..constraints.transforms import (
    EGARCHConstraint,
    GARCHConstraint,
    GJRConstraint,
    inverse_positive,
)
from ..core.config import NumericalConfig
from .base import CellState, InitMethod, VarianceCell


class GARCHCell(VarianceCell):
    r"""The GARCH(1,1) recursion as an autograd-tracked recurrent cell.

    .. math::
        h_t = \omega + \alpha \epsilon_{t-1}^2 + \beta h_{t-1}

    Zhao et al. (2024) observe that this is exactly a recurrent cell with a
    scalar hidden state, no output layer and no activation: the parameter vector
    is :math:`\Theta = (\omega, \alpha, \beta)`, the observation vector is
    :math:`X = (1, \epsilon_{t-1}^2, h_{t-1})`, and :math:`h_t = \Theta \cdot X`.
    Writing it as an ``nn.Module`` is what makes that equivalence usable: the
    same object can be estimated by maximum likelihood *or* dropped inside a
    larger network and trained end to end.

    Parameters
    ----------
    omega, alpha, beta
        Starting values on the **econometric** scale. They are inverted through
        the constraint transform to initialise the raw parameters, so a user
        never has to think in unconstrained space.
    enforce_stationarity
        When ``True`` (default), a softmax simplex guarantees
        :math:`\alpha + \beta < 1` for every optimiser step. When ``False``,
        only non-negativity is imposed, which is what you want if you are
        deliberately fitting an IGARCH-like process.

    Examples
    --------
    >>> import torch
    >>> _ = torch.manual_seed(0)
    >>> cell = GARCHCell(omega=0.05, alpha=0.10, beta=0.85)
    >>> p = {k: v.detach() for k, v in cell.transformed_parameters().items()}
    >>> round(float(p["alpha"]), 4), round(float(p["beta"]), 4)
    (0.1, 0.85)
    >>> eps = torch.randn(500, dtype=torch.float64)
    >>> h, _ = cell.filter(eps)
    >>> h.shape, bool((h > 0).all())
    (torch.Size([500]), True)
    """

    constraint_family = "garch"

    def __init__(
        self,
        omega: float = 0.05,
        alpha: float = 0.10,
        beta: float = 0.85,
        enforce_stationarity: bool = True,
        init_method: InitMethod = "arch_backcast",
        backcast_lambda: float | None = None,
        sample_n: int = 10,
        numerical: NumericalConfig | None = None,
    ):
        super().__init__(
            init_method=init_method,
            backcast_lambda=backcast_lambda,
            sample_n=sample_n,
            numerical=numerical,
        )
        self.constraint = GARCHConstraint(
            slack=self.numerical.stationarity_slack,
            eps=self.numerical.eps,
            enforce=enforce_stationarity,
        )
        raw = _invert_garch_start(omega, alpha, beta, self.constraint)
        self.raw_omega = nn.Parameter(raw["omega"])
        self.raw_alpha = nn.Parameter(raw["alpha"])
        self.raw_beta = nn.Parameter(raw["beta"])
        self.raw_slack = nn.Parameter(raw["slack"])

    def transformed_parameters(self) -> dict[str, Tensor]:
        return self.constraint(self.raw_omega, self.raw_alpha, self.raw_beta, self.raw_slack)

    def step(
        self, eps_prev: Tensor, state: CellState, params: dict[str, Tensor] | None = None
    ) -> tuple[Tensor, CellState]:
        p = params if params is not None else self.transformed_parameters()
        h = p["omega"] + p["alpha"] * eps_prev.square() + p["beta"] * state.h
        h = h.clamp(min=self.numerical.variance_floor)
        return h, CellState(h=h)

    def affine_terms(self, eps: Tensor, params: dict[str, Tensor]):
        """``h_t = (omega + alpha * eps_{t-1}^2) + beta * h_{t-1}`` - affine in h."""
        return params["omega"] + params["alpha"] * eps.square(), params["beta"]

    def h1_from_backcast(self, backcast: Tensor) -> Tensor:
        """``arch`` convention: ``h_1 = omega + (alpha + beta) * backcast``."""
        p = self.transformed_parameters()
        return p["omega"] + (p["alpha"] + p["beta"]) * backcast

    def persistence(self) -> Tensor:
        return self.constraint.persistence(self.transformed_parameters())


class GJRCell(VarianceCell):
    r"""GJR-GARCH(1,1,1), the asymmetric-response recursion.

    .. math::
        h_t = \omega + \alpha \epsilon_{t-1}^2
              + \gamma\, I(\epsilon_{t-1} < 0)\, \epsilon_{t-1}^2
              + \beta h_{t-1}

    ``gamma > 0`` is the leverage effect: a negative shock of a given size
    raises tomorrow's variance more than a positive one. Zhao et al. (2024) use
    GJR as the kernel of their best-performing GARCH-LSTM, precisely because the
    asymmetry is the stylized fact a plain LSTM has the hardest time inferring.

    Note that the cell takes :math:`\epsilon_{t-1}` rather than
    :math:`\epsilon_{t-1}^2` as its input, because the *sign* is needed.
    """

    constraint_family = "gjr"

    def __init__(
        self,
        omega: float = 0.05,
        alpha: float = 0.05,
        gamma: float = 0.08,
        beta: float = 0.85,
        enforce_stationarity: bool = True,
        init_method: InitMethod = "arch_backcast",
        backcast_lambda: float | None = None,
        sample_n: int = 10,
        numerical: NumericalConfig | None = None,
    ):
        super().__init__(
            init_method=init_method,
            backcast_lambda=backcast_lambda,
            sample_n=sample_n,
            numerical=numerical,
        )
        self.constraint = GJRConstraint(
            slack=self.numerical.stationarity_slack,
            eps=self.numerical.eps,
            enforce=enforce_stationarity,
        )
        raw = _invert_gjr_start(omega, alpha, gamma, beta, self.constraint)
        self.raw_omega = nn.Parameter(raw["omega"])
        self.raw_alpha = nn.Parameter(raw["alpha"])
        self.raw_gamma = nn.Parameter(raw["gamma"])
        self.raw_beta = nn.Parameter(raw["beta"])
        self.raw_slack = nn.Parameter(raw["slack"])

    def transformed_parameters(self) -> dict[str, Tensor]:
        return self.constraint(
            self.raw_omega, self.raw_alpha, self.raw_gamma, self.raw_beta, self.raw_slack
        )

    def step(
        self, eps_prev: Tensor, state: CellState, params: dict[str, Tensor] | None = None
    ) -> tuple[Tensor, CellState]:
        p = params if params is not None else self.transformed_parameters()
        sq = eps_prev.square()
        indicator = (eps_prev < 0).to(sq.dtype)
        h = p["omega"] + p["alpha"] * sq + p["gamma"] * indicator * sq + p["beta"] * state.h
        h = h.clamp(min=self.numerical.variance_floor)
        return h, CellState(h=h)

    def affine_terms(self, eps: Tensor, params: dict[str, Tensor]):
        """Affine in ``h``: the indicator depends on ``eps``, not on the variance."""
        sq = eps.square()
        indicator = (eps < 0).to(sq.dtype)
        c = params["omega"] + params["alpha"] * sq + params["gamma"] * indicator * sq
        return c, params["beta"]

    def h1_from_backcast(self, backcast: Tensor) -> Tensor:
        """``arch`` convention: ``h_1 = omega + (alpha + gamma/2 + beta) * backcast``.

        The ``gamma/2`` reflects ``E[I(eps < 0)] = 1/2`` under symmetric
        innovations, so the pre-sample asymmetric term contributes half its
        coefficient.
        """
        p = self.transformed_parameters()
        return p["omega"] + (p["alpha"] + 0.5 * p["gamma"] + p["beta"]) * backcast

    def persistence(self) -> Tensor:
        return self.constraint.persistence(self.transformed_parameters())


@torch.jit.script
def _egarch_log_variance_scan(
    eps: Tensor,
    log_h1: Tensor,
    omega: Tensor,
    alpha: Tensor,
    gamma: Tensor,
    beta: Tensor,
    expected_abs_z: float,
    log_floor: float,
    log_ceiling: float,
) -> Tensor:
    """TorchScript inner loop for the EGARCH log-variance recursion.

    EGARCH cannot use the vectorised affine path, because
    :math:`z_{t-1} = \\epsilon_{t-1}/\\sqrt{h_{t-1}}` makes the recursion
    nonlinear in the lagged variance. Compiling the scalar loop removes the
    Python and dispatch overhead instead, which is where essentially all of the
    cost was.
    """
    n = eps.shape[0]
    outputs = [log_h1]
    log_h = log_h1
    for t in range(1, n):
        z = eps[t - 1] * torch.exp(-0.5 * log_h)
        log_h = (
            omega
            + beta * log_h
            + alpha * (torch.abs(z) - expected_abs_z)
            + gamma * z
        )
        log_h = torch.clamp(log_h, log_floor, log_ceiling)
        outputs.append(log_h)
    return torch.stack(outputs)


class EGARCHCell(VarianceCell):
    r"""EGARCH(1,1,1), Nelson's exponential specification.

    .. math::
        \log h_t = \omega + \beta \log h_{t-1}
                   + \alpha\big(|z_{t-1}| - E|z|\big) + \gamma z_{t-1},
        \qquad z_{t-1} = \epsilon_{t-1}/\sqrt{h_{t-1}}

    Modelling the log variance makes positivity structural rather than
    constrained, which is why EGARCH is the most convenient family to
    neuralise: a parameter network can emit unbounded reals and the variance is
    still valid. The price is that the recursion can overflow, so
    :math:`\log h_t` is clipped at ``numerical.max_log_variance``.

    Parameters
    ----------
    expected_abs_z
        :math:`E|z|` under the innovation distribution. Defaults to
        :math:`\sqrt{2/\pi}`, the standard normal value. For a standardised
        Student-t this differs, and leaving it at the normal value biases
        ``omega``; pass the correct constant when the distribution is not normal.
    """

    constraint_family = "egarch"

    def __init__(
        self,
        omega: float = -0.05,
        alpha: float = 0.15,
        gamma: float = -0.08,
        beta: float = 0.97,
        expected_abs_z: float = math.sqrt(2.0 / math.pi),
        enforce_stationarity: bool = True,
        init_method: InitMethod = "arch_backcast",
        backcast_lambda: float | None = None,
        sample_n: int = 10,
        numerical: NumericalConfig | None = None,
    ):
        super().__init__(
            init_method=init_method,
            backcast_lambda=backcast_lambda,
            sample_n=sample_n,
            numerical=numerical,
        )
        self.constraint = EGARCHConstraint(
            slack=self.numerical.stationarity_slack, enforce=enforce_stationarity
        )
        self.expected_abs_z = expected_abs_z
        dt = torch.float64
        self.raw_omega = nn.Parameter(torch.tensor(float(omega), dtype=dt))
        self.raw_alpha = nn.Parameter(torch.tensor(float(alpha), dtype=dt))
        self.raw_gamma = nn.Parameter(torch.tensor(float(gamma), dtype=dt))
        # Invert beta = (1 - slack) * tanh(raw). Dividing by the budget before
        # atanh is what makes the round-trip exact: without it the stored beta
        # comes back as (1 - slack) * beta, a 1e-4 relative error that is small
        # enough to hide and large enough to break bit-parity with `arch`.
        budget = 1.0 - self.numerical.stationarity_slack if enforce_stationarity else 1.0
        target = float(beta) / budget
        clipped = min(max(target, -0.999999), 0.999999)
        self.raw_beta = nn.Parameter(torch.atanh(torch.tensor(clipped, dtype=dt)))

    def transformed_parameters(self) -> dict[str, Tensor]:
        return self.constraint(self.raw_omega, self.raw_alpha, self.raw_gamma, self.raw_beta)

    def step(
        self, eps_prev: Tensor, state: CellState, params: dict[str, Tensor] | None = None
    ) -> tuple[Tensor, CellState]:
        p = params if params is not None else self.transformed_parameters()
        log_h_prev = state.log_h if state.log_h is not None else torch.log(state.h)
        z = eps_prev * torch.rsqrt(state.h)
        log_h = (
            p["omega"]
            + p["beta"] * log_h_prev
            + p["alpha"] * (z.abs() - self.expected_abs_z)
            + p["gamma"] * z
        )
        log_h = log_h.clamp(
            min=math.log(self.numerical.variance_floor), max=self.numerical.max_log_variance
        )
        h = torch.exp(log_h)
        return h, CellState(h=h, log_h=log_h)

    def filter(
        self,
        eps: Tensor,
        state: CellState | None = None,
        truncate_bptt: int | None = None,
    ) -> tuple[Tensor, CellState]:
        """Filter via the compiled log-variance scan.

        Falls back to the generic step loop when truncated BPTT is requested,
        since that deliberately cuts the graph at intervals the scan knows
        nothing about.
        """
        if truncate_bptt is not None:
            return super().filter(eps, state=state, truncate_bptt=truncate_bptt)
        if eps.ndim != 1:
            raise ValueError(
                f"filter expects a 1-D innovation series, got shape {tuple(eps.shape)}."
            )
        if state is None:
            state = self.initial_state(eps)
        p = self.transformed_parameters()
        log_h1 = state.log_h if state.log_h is not None else torch.log(state.h)
        log_h = _egarch_log_variance_scan(
            eps,
            log_h1,
            p["omega"],
            p["alpha"],
            p["gamma"],
            p["beta"],
            float(self.expected_abs_z),
            math.log(self.numerical.variance_floor),
            float(self.numerical.max_log_variance),
        )
        h = torch.exp(log_h)
        return h, CellState(h=h[-1], log_h=log_h[-1])

    def h1_from_backcast(self, backcast: Tensor) -> Tensor:
        r"""``arch`` convention, applied on the log scale.

        ``arch`` backcasts :math:`\log h_0 = \log(\mathrm{bc})` and takes one
        step with the pre-sample innovation at its mean, so both
        :math:`|z| - E|z|` and :math:`z` vanish, leaving
        :math:`\log h_1 = \omega + \beta \log(\mathrm{bc})`.
        """
        p = self.transformed_parameters()
        log_h1 = p["omega"] + p["beta"] * torch.log(backcast)
        return torch.exp(log_h1.clamp(max=self.numerical.max_log_variance))

    def persistence(self) -> Tensor:
        return self.constraint.persistence(self.transformed_parameters())

    def unconditional_variance(self) -> Tensor:
        r"""Long-run variance :math:`\exp\!\big(\omega/(1-\beta)\big)`.

        EGARCH's long-run level lives on the log scale, so the generic
        :math:`\omega/(1-P)` formula does not apply.
        """
        p = self.transformed_parameters()
        denom = 1.0 - p["beta"]
        if float(denom) <= 0:
            return torch.tensor(float("inf"), dtype=p["omega"].dtype)
        return torch.exp((p["omega"] / denom).clamp(max=self.numerical.max_log_variance))


# --------------------------------------------------------------------------- #
# helpers: econometric starting values -> raw parameters
# --------------------------------------------------------------------------- #
def _invert_garch_start(
    omega: float, alpha: float, beta: float, constraint: GARCHConstraint
) -> dict[str, Tensor]:
    """Find raw parameters whose transform equals the requested ``(w, a, b)``.

    The softmax simplex is invertible up to an additive constant: setting the
    raw logits to ``log(w_i)`` reproduces the weights exactly. We solve for the
    slack weight as the residual ``1 - alpha - beta`` (on the ``1 - delta``
    budget), so a user asking for ``alpha=0.1, beta=0.85`` gets exactly that.
    """
    dt = torch.float64
    if not constraint.enforce:
        return {
            "omega": inverse_positive(omega, constraint.eps).to(dt),
            "alpha": inverse_positive(max(alpha, 1e-8), 0.0).to(dt),
            "beta": inverse_positive(max(beta, 1e-8), 0.0).to(dt),
            "slack": torch.zeros((), dtype=dt),
        }
    budget = 1.0 - constraint.slack
    total = alpha + beta
    if total >= budget:
        raise ValueError(
            f"alpha + beta = {total:.6f} violates stationarity (must be < {budget:.6f}). "
            "Pass enforce_stationarity=False to fit a non-stationary process."
        )
    weights = torch.tensor(
        [alpha / budget, beta / budget, 1.0 - total / budget], dtype=dt
    ).clamp(min=1e-12)
    logits = torch.log(weights)
    return {
        "omega": inverse_positive(omega, constraint.eps).to(dt),
        "alpha": logits[0],
        "beta": logits[1],
        "slack": logits[2],
    }


def _invert_gjr_start(
    omega: float, alpha: float, gamma: float, beta: float, constraint: GJRConstraint
) -> dict[str, Tensor]:
    """Raw parameters reproducing ``(omega, alpha, gamma, beta)`` for GJR."""
    dt = torch.float64
    if not constraint.enforce:
        return {
            "omega": inverse_positive(omega, constraint.eps).to(dt),
            "alpha": inverse_positive(max(alpha, 1e-8), 0.0).to(dt),
            "gamma": torch.tensor(float(gamma), dtype=dt),
            "beta": inverse_positive(max(beta, 1e-8), 0.0).to(dt),
            "slack": torch.zeros((), dtype=dt),
        }
    if gamma < 0:
        raise ValueError(
            "The stationary GJR parameterisation requires gamma >= 0 (the usual "
            "leverage direction). Pass enforce_stationarity=False for gamma < 0."
        )
    budget = 1.0 - constraint.slack
    total = alpha + 0.5 * gamma + beta
    if total >= budget:
        raise ValueError(
            f"alpha + gamma/2 + beta = {total:.6f} violates stationarity "
            f"(must be < {budget:.6f})."
        )
    weights = torch.tensor(
        [alpha / budget, 0.5 * gamma / budget, beta / budget, 1.0 - total / budget], dtype=dt
    ).clamp(min=1e-12)
    logits = torch.log(weights)
    return {
        "omega": inverse_positive(omega, constraint.eps).to(dt),
        "alpha": logits[0],
        "gamma": logits[1],
        "beta": logits[2],
        "slack": logits[3],
    }
