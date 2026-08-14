r"""Yin & Barucca (2022): Neural Generalised ARCH.

The GARCH coefficients stop being constants and become **latent random
variables that move with time**:

.. math::
    \sigma_t^2 &= \omega_t + \alpha_t r_{t-1}^2 + \beta_t\sigma_{t-1}^2
        \qquad\text{(Eq. 10)} \\
    \Sigma_t &= \Omega_t^{\!\top}\Omega_t
        + A_t^{\!\top} r_{t-1}r_{t-1}^{\!\top}A_t
        + B_t^{\!\top}\Sigma_{t-1}B_t \qquad\text{(Eq. 11)}

with :math:`\gamma_t = (\omega_t,\alpha_t,\beta_t)^{\!\top}` drawn from a
diagonal Gaussian whose moments are produced by a network. Inference is
amortised variational Bayes in the VRNN style: a GRU carries the return history,
and two MLPs sharing it produce a *prior* and a *posterior* over
:math:`\gamma_t`,

.. math::
    p_\theta(\gamma_t\mid\gamma_{t-1}, h_{t-1}), \qquad
    q_\phi(\gamma_t\mid\gamma_{t-1}, h_t),

trained jointly on the ELBO (Eq. 21). The distinction between the two is only
*which* hidden state they see: the prior predicts :math:`\gamma_t` from
information through :math:`t-1`, the posterior updates it having seen
:math:`r_t`. That asymmetry is what makes one-step-ahead forecasting
well defined — see :meth:`NeuralGARCH.forecast_path`.

Why keep the GARCH form at all, if a network is generating the coefficients?
Because it buys structure for free. In the multivariate case, preserving the
BEKK quadratic form means :math:`\Sigma_t` is symmetric positive definite by
construction, with no projection step and no eigenvalue repair — exactly the
guarantee blueprint sec. 18 demands.

Two deviations from the printed paper, both recorded on the provenance
----------------------------------------------------------------------
1. **Equation 4 has a sign error.** The Student-t log-likelihood is printed with
   ``+ log Gamma(nu/2)`` inside a quantity being maximised; the standardised
   Student-t density has ``- log Gamma(nu/2)``. We use the mathematically
   correct form via :class:`~garchai.distributions.StudentT`.
2. **Stationarity is not enforced**, following the paper. A ``sigmoid`` bounds
   each coefficient to :math:`(0,1)` individually, so :math:`\alpha_t+\beta_t`
   may exceed one — which the authors argue is realistic, citing evidence that
   relaxing stationarity improves fit. ``enforce_stationarity=True`` switches to
   a simplex if you want the classical restriction back.

Reference
---------
Yin, Z., & Barucca, P. (2022). Neural Generalised AutoRegressive Conditional
Heteroskedasticity. https://arxiv.org/abs/2202.11285
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch import Tensor

from ..constraints.transforms import stationary_simplex
from ..core.base import VolatilityModel
from ..core.config import ModelConfig, TrainConfig
from ..core.provenance import CodeStatus, ImplementationStatus, Provenance
from ..core.results import FitResult, to_numpy, to_tensor
from ..distributions import Normal, StudentT
from ..distributions.base import Distribution
from ..mean.models import MeanModel, get_mean_model

__all__ = ["NeuralGARCH", "NeuralBEKK"]

_YIN = dict(
    paper_id="yin2022_neural_garch",
    title="Neural Generalised AutoRegressive Conditional Heteroskedasticity",
    authors=("Zexuan Yin", "Paolo Barucca"),
    year=2022,
    paper_url="https://arxiv.org/abs/2202.11285",
    code_status=CodeStatus.NO_PUBLIC_CODE_IDENTIFIED,
    implementation_status=ImplementationStatus.PAPER_FAITHFUL,
)

_DEVIATIONS = (
    "Eq. 4 prints the Student-t log-likelihood with +logGamma(nu/2); the "
    "standardised density requires -logGamma(nu/2). The correct form is used.",
    "Stationarity is not enforced, following the paper: a sigmoid bounds each "
    "coefficient to (0,1) individually, so alpha_t + beta_t may exceed 1. "
    "enforce_stationarity=True restores the classical simplex.",
    "The community repository stuid-maker/neural-garch is NOT this model and "
    "was not used as a reference.",
    "gamma_t is a Gaussian *draw*, so the paper's sigmoid bounds its moments "
    "but not the realisation: a sampled alpha_t or beta_t can be negative or "
    "exceed 1. We clamp the draw at zero, since a negative coefficient would "
    "make the 'variance' negative, and cap the variance path at "
    "exp(max_log_variance) so an explosive sampled beta_t cannot overflow.",
)


def _mlp(in_features: int, out_features: int, hidden: int = 64, layers: int = 3) -> nn.Sequential:
    """The paper's parameter network: 3 layers, 64 units, ReLU."""
    blocks: list[nn.Module] = []
    size = in_features
    for _ in range(layers - 1):
        blocks += [nn.Linear(size, hidden, dtype=torch.float64), nn.ReLU()]
        size = hidden
    blocks.append(nn.Linear(size, out_features, dtype=torch.float64))
    return nn.Sequential(*blocks)


def _gaussian_kl(
    mu_q: Tensor, var_q: Tensor, mu_p: Tensor, var_p: Tensor, eps: float = 1e-8
) -> Tensor:
    r"""KL divergence between two diagonal Gaussians, summed over dimensions.

    .. math::
        \mathrm{KL} = \tfrac{1}{2}\sum_i\left[
            \log\frac{\sigma^2_{p,i}}{\sigma^2_{q,i}}
            + \frac{\sigma^2_{q,i} + (\mu_{q,i}-\mu_{p,i})^2}{\sigma^2_{p,i}} - 1
        \right]
    """
    var_q = var_q.clamp(min=eps)
    var_p = var_p.clamp(min=eps)
    return 0.5 * (
        torch.log(var_p / var_q) + (var_q + (mu_q - mu_p).square()) / var_p - 1.0
    ).sum()


class _CoefficientNetwork(nn.Module):
    """The shared GRU plus the prior and posterior MLPs (paper sec. 3.2-3.3)."""

    def __init__(
        self,
        n_coefficients: int,
        hidden_size: int = 64,
        mlp_hidden: int = 64,
        mean_init: tuple[float, ...] | None = None,
        variance_init: float = 1e-3,
    ):
        super().__init__()
        self.n_coefficients = n_coefficients
        self.hidden_size = hidden_size
        self.gru = nn.GRUCell(1, hidden_size, dtype=torch.float64)
        # Both MLPs read (gamma_{t-1}, h) and emit means and variances.
        self.prior = _mlp(n_coefficients + hidden_size, 2 * n_coefficients, mlp_hidden)
        self.posterior = _mlp(n_coefficients + hidden_size, 2 * n_coefficients, mlp_hidden)
        self._initialise_output_bias(mean_init, variance_init)

    def _initialise_output_bias(
        self, mean_init: tuple[float, ...] | None, variance_init: float
    ) -> None:
        r"""Start the coefficient distribution somewhere econometrically sensible.

        With default initialisation every sigmoid output sits near 0.5, so the
        recursion starts at :math:`\beta_t \approx 0.5` with a *standard
        deviation* of roughly 0.7 on each coefficient. Sampled
        :math:`\beta_t > 1` then makes the variance explode, and the ELBO opens
        at :math:`10^7` before training has had a chance to do anything.

        Setting the final-layer bias so the prior begins at
        :math:`(\omega,\alpha,\beta)\approx(0.05, 0.1, 0.85)` with a small
        variance fixes that. This changes only the starting point, not the model.
        """
        if mean_init is None:
            defaults = [0.05, 0.10, 0.85, 0.06]  # omega, alpha, beta, nu' scale
            mean_init = tuple(defaults[: self.n_coefficients])
        if len(mean_init) != self.n_coefficients:
            raise ValueError(
                f"mean_init has {len(mean_init)} entries but the coefficient "
                f"vector has {self.n_coefficients}."
            )

        def logit(x: float) -> float:
            x = min(max(x, 1e-6), 1.0 - 1e-6)
            return float(torch.logit(torch.tensor(x, dtype=torch.float64)))

        bias = torch.tensor(
            [logit(v) for v in mean_init] + [logit(variance_init)] * self.n_coefficients,
            dtype=torch.float64,
        )
        for network in (self.prior, self.posterior):
            final = network[-1]
            with torch.no_grad():
                final.bias.copy_(bias)
                final.weight.mul_(0.01)  # start near the bias, then learn away from it

    def _split(self, raw: Tensor) -> tuple[Tensor, Tensor]:
        mean_raw, var_raw = raw.chunk(2, dim=-1)
        # The paper applies a sigmoid to keep both the coefficients and their
        # variances non-negative, and reports it outperformed softplus.
        return torch.sigmoid(mean_raw), torch.sigmoid(var_raw)

    def prior_moments(self, gamma_prev: Tensor, h_prev: Tensor) -> tuple[Tensor, Tensor]:
        return self._split(self.prior(torch.cat([gamma_prev, h_prev])))

    def posterior_moments(self, gamma_prev: Tensor, h_now: Tensor) -> tuple[Tensor, Tensor]:
        return self._split(self.posterior(torch.cat([gamma_prev, h_now])))

    def advance(self, r_t: Tensor, h: Tensor) -> Tensor:
        return self.gru(r_t.reshape(1, 1), h.reshape(1, -1)).reshape(-1)


class _NeuralGARCHBase(VolatilityModel):
    """Shared ELBO machinery for the univariate and multivariate variants."""

    def __init__(self, config: ModelConfig | None = None, hidden_size: int = 64):
        super().__init__(config)
        self._mean = get_mean_model(self.config.mean)
        self.hidden_size = hidden_size
        self.kl_weight = 1.0
        # Sampled beta_t is unbounded above, so the recursion needs a ceiling.
        self._variance_ceiling = float(
            torch.exp(torch.tensor(self.config.numerical.max_log_variance))
        )

    @property
    def mean_model(self) -> MeanModel:
        return self._mean

    @property
    def cell(self):
        return self.network

    def supports_analytic_forecast(self) -> bool:
        return False

    def _sample(self, mean: Tensor, variance: Tensor) -> Tensor:
        """Reparameterised draw, so gradients flow through the sample."""
        if self.training:
            return mean + torch.sqrt(variance.clamp(min=1e-12)) * torch.randn_like(mean)
        return mean

    def _variance_diagnostics(self) -> tuple[dict[str, float], float, float]:
        """Report the *average* coefficient path — they are time-varying here."""
        if getattr(self, "_last_paths", None) is None:
            return {}, float("nan"), float("nan")
        paths = self._last_paths
        params = {f"mean {k}": float(v.mean()) for k, v in paths.items()}
        persistence = float((paths["alpha"] + paths["beta"]).mean())
        omega_bar = float(paths["omega"].mean())
        long_run = omega_bar / (1.0 - persistence) if persistence < 1.0 else float("inf")
        return params, persistence, long_run


class NeuralGARCH(_NeuralGARCHBase):
    r"""Univariate Neural GARCH(1,1) with time-varying coefficients.

    Parameters
    ----------
    hidden_size
        GRU hidden width. The paper uses 64.
    distribution
        ``"normal"`` or ``"student-t"``. Under the t, the degrees of freedom is
        itself time-varying: the paper augments :math:`\gamma_t` with
        :math:`\nu_t` and estimates :math:`\nu' = \nu - 2` so that
        :math:`\nu > 2` and the variance exists.
    enforce_stationarity
        ``False`` reproduces the paper. ``True`` applies a simplex so that
        :math:`\alpha_t + \beta_t < 1` at every step.

    Attributes
    ----------
    parameter_paths_
        After fitting, a dict of arrays ``omega``, ``alpha``, ``beta`` (and
        ``nu`` under the t) giving the estimated coefficient path. This is the
        model's main interpretive payoff: you can *watch* the persistence move
        through a crisis instead of assuming it constant.

    Examples
    --------
    >>> import torch
    >>> torch.set_default_dtype(torch.float64)
    >>> from garchai.core import set_seed
    >>> set_seed(0)
    >>> model = NeuralGARCH(hidden_size=8)
    >>> h, eps, _ = model.filter(torch.randn(60) * 0.9)
    >>> h.shape, bool((h > 0).all())
    (torch.Size([60]), True)
    """

    provenance = Provenance(
        **_YIN,
        equations="Eqs. 10, 12, 16-17, 19-21 (univariate)",
        deviations=_DEVIATIONS,
    )

    def __init__(
        self,
        hidden_size: int = 64,
        config: ModelConfig | None = None,
        enforce_stationarity: bool = False,
        mlp_hidden: int = 64,
    ):
        super().__init__(config, hidden_size)
        self.enforce_stationarity = enforce_stationarity
        self.use_student_t = self.config.distribution == "student-t"
        if self.config.distribution not in ("normal", "student-t"):
            raise ValueError("NeuralGARCH supports 'normal' or 'student-t'.")

        # gamma_t = (omega, alpha, beta) and, under the t, nu' = nu - 2.
        self.n_coefficients = 4 if self.use_student_t else 3
        self.network = _CoefficientNetwork(self.n_coefficients, hidden_size, mlp_hidden)
        self._last_paths: dict | None = None
        self._last_kl: Tensor | None = None

    def distribution(self) -> Distribution:
        """A placeholder; the real per-step distribution is built inside the filter."""
        return StudentT(df=8.0) if self.use_student_t else Normal()

    def _coefficients(self, gamma: Tensor) -> dict[str, Tensor]:
        """Map a raw draw to valid GARCH coefficients."""
        omega, alpha, beta = gamma[0], gamma[1], gamma[2]
        if self.enforce_stationarity:
            alpha, beta = stationary_simplex(
                torch.logit(alpha.clamp(1e-6, 1 - 1e-6)),
                torch.logit(beta.clamp(1e-6, 1 - 1e-6)),
                torch.zeros_like(alpha),
                self.config.numerical.stationarity_slack,
            )
        out = {"omega": omega.clamp(min=self.config.numerical.eps), "alpha": alpha, "beta": beta}
        if self.use_student_t:
            # nu' = nu - 2, scaled into a usable range so that nu > 2 always.
            out["nu"] = 2.0 + 100.0 * gamma[3].clamp(min=1e-4)
        return out

    def filter(self, r: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Run the generative/inference recursion, accumulating the KL term."""
        eps, mu = self.mean_model.residuals(r)
        n = eps.numel()
        dtype = eps.dtype

        h = torch.zeros(self.hidden_size, dtype=dtype)
        # P(gamma_0) is a delta on a vector of ones (paper sec. 3.2).
        gamma = torch.ones(self.n_coefficients, dtype=dtype)
        sigma2 = eps.var(unbiased=True).clamp(min=self.config.numerical.variance_floor)

        variances = [sigma2]
        paths: dict[str, list[Tensor]] = {"omega": [], "alpha": [], "beta": []}
        if self.use_student_t:
            paths["nu"] = []
        kl_total = torch.zeros((), dtype=dtype)

        for t in range(1, n):
            # The prior sees information through t-1; the posterior also sees r_t.
            mu_p, var_p = self.network.prior_moments(gamma, h)
            h = self.network.advance(eps[t - 1], h)
            mu_q, var_q = self.network.posterior_moments(gamma, h)

            kl_total = kl_total + _gaussian_kl(mu_q, var_q, mu_p, var_p)
            gamma = self._sample(mu_q, var_q).clamp(min=0.0)

            coefficients = self._coefficients(gamma)
            sigma2 = (
                coefficients["omega"]
                + coefficients["alpha"] * eps[t - 1].square()
                + coefficients["beta"] * sigma2
            ).clamp(
                min=self.config.numerical.variance_floor,
                max=self._variance_ceiling,
            )

            variances.append(sigma2)
            for key in paths:
                paths[key].append(coefficients[key])

        self._last_paths = {k: torch.stack(v).detach().cpu().numpy() for k, v in paths.items()}
        self._last_kl = kl_total
        self._last_nu = torch.stack(paths["nu"]) if self.use_student_t else None
        return torch.stack(variances), eps, mu

    def negative_loglikelihood(self, r: Tensor) -> Tensor:
        r"""The negative ELBO (Eq. 21), not a plain likelihood.

        .. math::
            \text{ELBO} = \sum_t E_{q}\big[\log p(r_t\mid\gamma_t)\big]
                        - \mathrm{KL}\big(q_\phi \,\|\, p_\theta\big)

        Reported as a negative so the training loop can minimise it uniformly.
        Note that it is a *bound*, not the marginal likelihood, so AIC and BIC
        computed from it are not comparable with those of a plain MLE model —
        the summary reports it as an ELBO for that reason.
        """
        h, eps, _ = self.filter(r)
        start = self.mean_model.n_presample

        if self.use_student_t:
            # nu_t varies with t, so the density is evaluated step by step.
            nu = self._last_nu
            z = eps[1:] * torch.rsqrt(h[1:])
            log_prob = torch.stack(
                [StudentT(df=nu[i]).log_prob(z[i]) for i in range(z.numel())]
            )
            log_likelihood = (log_prob - 0.5 * torch.log(h[1:])).sum()
        else:
            log_likelihood = -Normal().nll(eps[start:], h[start:], reduction="sum")

        return -(log_likelihood - self.kl_weight * self._last_kl)

    def _build_result(self, r: Tensor, history: dict, message: str) -> FitResult:
        result = super()._build_result(r, history, message)
        result.parameter_paths = self._last_paths
        result.message = message + "; loglikelihood is an ELBO (variational bound)"
        return result

    def forecast_path(self, r: Tensor, horizon: int = 1) -> Tensor:
        r"""One-step-ahead variance forecasts using the *prior* network.

        The paper's Section 3.5 is precise about the mechanics, and they matter.
        To predict :math:`\Sigma_{t+1}` we draw :math:`\gamma_{t+1}` from the
        **prior** ``MLP_pred``, because at time :math:`t` we have not seen
        :math:`r_{t+1}`. Once :math:`r_{t+1}` arrives we replace that draw with
        the **posterior** estimate before stepping forward again. Skipping that
        posterior update would throw away the most recent observation at every
        step and quietly degrade the whole path.
        """
        if horizon != 1:
            raise NotImplementedError(
                "Neural GARCH produces one-step-ahead forecasts (paper sec. 3.5). "
                "For multi-step, simulate the coefficient process explicitly."
            )
        self.eval()
        with torch.no_grad():
            eps, _ = self.mean_model.residuals(r)
            h = torch.zeros(self.hidden_size, dtype=eps.dtype)
            gamma = torch.ones(self.n_coefficients, dtype=eps.dtype)
            sigma2 = eps.var(unbiased=True)

            for t in range(1, eps.numel()):
                h = self.network.advance(eps[t - 1], h)
                mu_q, _ = self.network.posterior_moments(gamma, h)
                gamma = mu_q
                c = self._coefficients(gamma)
                sigma2 = c["omega"] + c["alpha"] * eps[t - 1].square() + c["beta"] * sigma2

            mu_p, _ = self.network.prior_moments(gamma, h)   # prior: no r_{t+1} yet
            c = self._coefficients(mu_p)
            return c["omega"] + c["alpha"] * eps[-1].square() + c["beta"] * sigma2


class NeuralBEKK(_NeuralGARCHBase):
    r"""Multivariate Neural GARCH: a diagonal BEKK with neural coefficients.

    .. math::
        \Sigma_t = \Omega_t^{\!\top}\Omega_t
                 + A_t^{\!\top} r_{t-1}r_{t-1}^{\!\top} A_t
                 + B_t^{\!\top}\Sigma_{t-1}B_t

    with :math:`\Omega_t` upper triangular and :math:`A_t, B_t` diagonal.

    The quadratic form is doing real work. :math:`\Omega^{\!\top}\Omega` is
    positive semi-definite for *any* :math:`\Omega`, and the other two terms are
    congruence transforms of PSD matrices, so :math:`\Sigma_t` is symmetric
    positive definite by construction — whatever the network emits, at every
    step, with no projection, no eigenvalue clipping and no repair. That is why
    the paper keeps the BEKK structure instead of letting a network output a
    covariance matrix directly.

    For :math:`n` assets, :math:`\gamma_t` has :math:`2n + n(n+1)/2` entries:
    the diagonals of :math:`A` and :math:`B`, and the upper triangle of
    :math:`\Omega`.

    Examples
    --------
    >>> import torch
    >>> torch.set_default_dtype(torch.float64)
    >>> from garchai.core import set_seed
    >>> set_seed(0)
    >>> model = NeuralBEKK(n_assets=3, hidden_size=8)
    >>> cov = model.filter_covariance(torch.randn(40, 3) * 0.8)
    >>> cov.shape
    torch.Size([40, 3, 3])
    >>> eigenvalues = torch.linalg.eigvalsh(cov)
    >>> bool((eigenvalues > 0).all())          # positive definite at every step
    True
    """

    provenance = Provenance(
        **_YIN,
        equations="Eqs. 11, 13-14 (diagonal BEKK)",
        deviations=_DEVIATIONS,
    )

    def __init__(
        self,
        n_assets: int,
        hidden_size: int = 64,
        config: ModelConfig | None = None,
        mlp_hidden: int = 64,
    ):
        super().__init__(config, hidden_size)
        if n_assets < 2:
            raise ValueError("NeuralBEKK needs at least 2 assets; use NeuralGARCH otherwise.")
        # Eq. 5 specifies r_t ~ N(0, Sigma_t): the multivariate model is
        # zero-mean, so registering a constant-mean parameter would leave an
        # unused tensor in the optimiser.
        self._mean = get_mean_model("zero")
        self.n_assets = n_assets
        self.n_triangular = n_assets * (n_assets + 1) // 2
        self.n_coefficients = 2 * n_assets + self.n_triangular

        # BEKK enters the variance quadratically, so a_i and b_i start at the
        # square roots of the usual GARCH alpha and beta: a_i^2 ~ 0.10 and
        # b_i^2 ~ 0.85 reproduce a conventional persistence on the diagonal.
        mean_init = (
            [0.10**0.5] * n_assets            # diag(A)
            + [0.85**0.5] * n_assets          # diag(B)
            + [0.05] * self.n_triangular      # upper triangle of Omega
        )
        self.network = _CoefficientNetwork(
            self.n_coefficients, hidden_size, mlp_hidden, mean_init=tuple(mean_init)
        )
        # The GRU consumes the whole cross-section each step, not a scalar.
        self.network.gru = nn.GRUCell(n_assets, hidden_size, dtype=torch.float64)
        self._triu = torch.triu_indices(n_assets, n_assets)
        self._last_kl: Tensor | None = None

    def distribution(self) -> Distribution:
        return Normal()

    def _unpack(self, gamma: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Split the coefficient vector into ``(Omega, A, B)``."""
        n = self.n_assets
        a = torch.diag(gamma[:n])
        b = torch.diag(gamma[n : 2 * n])
        omega = torch.zeros(n, n, dtype=gamma.dtype)
        omega[self._triu[0], self._triu[1]] = gamma[2 * n :]
        return omega, a, b

    def filter_covariance(self, returns: Tensor) -> Tensor:
        """Return the covariance path, shape ``(T, n, n)``."""
        if returns.ndim != 2 or returns.shape[1] != self.n_assets:
            raise ValueError(
                f"returns must have shape (T, {self.n_assets}), got {tuple(returns.shape)}."
            )
        n_periods = returns.shape[0]
        dtype = returns.dtype

        h = torch.zeros(self.hidden_size, dtype=dtype)
        gamma = torch.ones(self.n_coefficients, dtype=dtype)
        sigma = torch.cov(returns.T) + 1e-6 * torch.eye(self.n_assets, dtype=dtype)

        covariances = [sigma]
        kl_total = torch.zeros((), dtype=dtype)

        for t in range(1, n_periods):
            mu_p, var_p = self.network.prior_moments(gamma, h)
            h = self.network.gru(
                returns[t - 1].reshape(1, -1), h.reshape(1, -1)
            ).reshape(-1)
            mu_q, var_q = self.network.posterior_moments(gamma, h)
            kl_total = kl_total + _gaussian_kl(mu_q, var_q, mu_p, var_p)
            gamma = self._sample(mu_q, var_q).clamp(min=0.0)

            omega, a, b = self._unpack(gamma)
            outer = torch.outer(returns[t - 1], returns[t - 1])
            sigma = omega.T @ omega + a.T @ outer @ a + b.T @ sigma @ b
            # A tiny ridge guards only against exact singularity when the
            # network emits an all-zero Omega; the PSD property itself is
            # structural and needs no help.
            sigma = sigma + 1e-10 * torch.eye(self.n_assets, dtype=dtype)
            covariances.append(sigma)

        self._last_kl = kl_total
        return torch.stack(covariances)

    def negative_loglikelihood(self, returns: Tensor) -> Tensor:
        """Negative ELBO under a multivariate normal likelihood."""
        covariances = self.filter_covariance(returns)
        cholesky = torch.linalg.cholesky(covariances)
        solved = torch.linalg.solve_triangular(
            cholesky, returns.unsqueeze(-1), upper=False
        ).squeeze(-1)
        quadratic = solved.square().sum(dim=-1)
        log_det = 2.0 * torch.log(torch.diagonal(cholesky, dim1=-2, dim2=-1)).sum(dim=-1)
        constant = self.n_assets * torch.log(torch.tensor(2.0 * math.pi, dtype=returns.dtype))
        log_likelihood = -0.5 * (constant + log_det + quadratic).sum()
        return -(log_likelihood - self.kl_weight * self._last_kl)

    def fit(self, returns, train: TrainConfig | None = None, callback=None) -> FitResult:
        """Fit by maximising the ELBO over the multivariate return panel."""
        cfg = train or TrainConfig()
        # NB: torch.Tensor has a `.values` method (the sparse accessor), so
        # `hasattr(x, "values")` is True for tensors and would hand a bound
        # method to as_tensor. Test for the pandas API instead.
        if hasattr(returns, "to_numpy"):
            returns = returns.to_numpy()
        data = torch.as_tensor(returns, dtype=self.config.torch_dtype)
        if data.ndim != 2:
            raise ValueError(f"NeuralBEKK expects a (T, n) panel, got {tuple(data.shape)}.")

        self.train()
        optimiser = self._build_optimiser(cfg)
        history: dict[str, list[float]] = {"loss": [], "val_loss": [], "lr": []}
        best, best_state, stalled = float("inf"), None, 0
        message = "maximum epochs reached"

        for epoch in range(cfg.max_epochs):
            optimiser.zero_grad(set_to_none=True)
            loss = self.negative_loglikelihood(data)
            if not torch.isfinite(loss):
                message = f"non-finite ELBO at epoch {epoch}; stopped"
                break
            loss.backward()
            if self.config.numerical.grad_clip is not None:
                nn.utils.clip_grad_norm_(self.parameters(), self.config.numerical.grad_clip)
            optimiser.step()

            value = float(loss.detach())
            history["loss"].append(value)
            history["val_loss"].append(float("nan"))
            history["lr"].append(optimiser.param_groups[0]["lr"])
            if value < best - 1e-10:
                best, stalled = value, 0
                best_state = {k: v.detach().clone() for k, v in self.state_dict().items()}
            else:
                stalled += 1
                if stalled >= cfg.patience:
                    message = f"early stopping at epoch {epoch}"
                    break
            if callback is not None:
                callback(epoch, value)

        if best_state is not None:
            self.load_state_dict(best_state)

        self.eval()
        with torch.no_grad():
            covariances = self.filter_covariance(data)
            variances = torch.diagonal(covariances, dim1=-2, dim2=-1)

        self.result_ = FitResult(
            conditional_variance=to_numpy(variances),
            conditional_volatility=to_numpy(torch.sqrt(variances)),
            residuals=to_numpy(data),
            standardized_residuals=to_numpy(data / torch.sqrt(variances)),
            conditional_mean=to_numpy(torch.zeros_like(data)),
            loglikelihood=-best,
            n_obs=int(data.shape[0]),
            n_params=sum(p.numel() for p in self.parameters()),
            n_presample=0,
            parameters={"n_assets": float(self.n_assets)},
            history=history,
            message=message + "; loglikelihood is an ELBO (variational bound)",
        )
        self._covariance_ = to_numpy(covariances)
        return self.result_

    @property
    def covariance_(self):
        """The fitted covariance path, shape ``(T, n, n)``."""
        if not hasattr(self, "_covariance_"):
            raise RuntimeError("Model has not been fitted yet; call .fit(returns) first.")
        return self._covariance_

    def correlation_(self):
        """The implied dynamic correlation path, shape ``(T, n, n)``."""
        import numpy as np

        cov = self.covariance_
        sd = np.sqrt(np.diagonal(cov, axis1=-2, axis2=-1))
        return cov / (sd[:, :, None] * sd[:, None, :])
