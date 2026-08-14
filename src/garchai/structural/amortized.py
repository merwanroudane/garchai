r"""De Clerk & Savel'ev (2022): AI algorithms for fitting GARCH parameters.

Every other model in this package *is* a volatility model. This one is an
**estimator** for a volatility model — a network that reads two summary
statistics off a return series and returns the GARCH parameters that would
generate them, in microseconds rather than the seconds a likelihood
optimisation takes.

The construction rests on two closed forms available for GARCH-normal(1,1). The
normalised autocovariance of squared returns at lag :math:`n` (Eq. 2),

.. math::
    \hat\gamma_n = \frac{2\alpha_1\big(1-\alpha_1\beta_1-\beta_1^2\big)}
                        {1-3\alpha_1^2-2\alpha_1\beta_1-\beta_1^2}
                   \,(\alpha_1+\beta_1)^{\,n-1},

and the fourth standardised moment (Eq. 3),

.. math::
    \Gamma_4 = 3 + \frac{6\alpha_1^2}
                        {1-3\alpha_1^2-2\alpha_1\beta_1-\beta_1^2}.

Together these map :math:`(\alpha_1,\beta_1) \mapsto (\hat\gamma_n,\Gamma_4)`.
The paper trains an MLP to invert that map, then recovers :math:`\beta_1` by
solving Eq. 3 and :math:`\alpha_0` from the unconditional variance (Eq. 4).

Why this is the one place synthetic data belongs
------------------------------------------------
The network's training set is 125,000 :math:`(\alpha_1,\beta_1)` draws pushed
through Eqs. 2-3. That is not a simulation study standing in for real data — it
*is* the function being learned, known in closed form. The evaluation still runs
on real assets. This is the sole exception to the package's real-data rule, and
it is an exception because the object of study is an analytic map.

The feasibility region is not optional
--------------------------------------
Not every :math:`(\Gamma_4, \hat\gamma_n)` pair is reachable by *any*
GARCH(1,1): the denominator :math:`1-3\alpha_1^2-2\alpha_1\beta_1-\beta_1^2` must
stay positive, which is exactly the condition for the fourth moment to exist.
The paper's Figure 1 maps the admissible region and discards empirical windows
that fall outside it — mostly short windows, where too few rare events have been
observed. :meth:`AmortizedGARCHInference.is_feasible` implements that check, and
:meth:`estimate` refuses rather than returning a fabricated parameter.

Reference
---------
De Clerk, L., & Savel'ev, S. (2022). AI algorithms for fitting GARCH parameters
to empirical financial data. *Physica A* 603, 127869.
https://doi.org/10.1016/j.physa.2022.127869 (CC BY-NC-ND)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

from ..core.config import TrainConfig, set_seed
from ..core.provenance import CodeStatus, ImplementationStatus, Provenance

__all__ = [
    "garch_moment_map",
    "normalised_autocovariance",
    "fourth_standardised_moment",
    "empirical_moments",
    "AmortizedGARCHInference",
    "NeuralGARCHEstimator",
    "GARCHParameters",
]


@dataclass
class GARCHParameters:
    """A recovered parameter triple, with the diagnostics behind it."""

    alpha0: float
    alpha1: float
    beta1: float
    feasible: bool
    persistence: float
    note: str = ""

    def as_dict(self) -> dict[str, float]:
        return {"omega": self.alpha0, "alpha": self.alpha1, "beta": self.beta1}


def _fourth_moment_denominator(alpha1, beta1):
    r""":math:`1 - 3\alpha_1^2 - 2\alpha_1\beta_1 - \beta_1^2`.

    Positive exactly when the fourth moment of the GARCH process exists. Both
    closed forms divide by it, so it governs the whole construction.
    """
    return 1.0 - 3.0 * alpha1**2 - 2.0 * alpha1 * beta1 - beta1**2


def normalised_autocovariance(alpha1, beta1, lag: int = 15):
    """Equation 2, the normalised autocovariance of squared returns."""
    denominator = _fourth_moment_denominator(alpha1, beta1)
    numerator = 2.0 * alpha1 * (1.0 - alpha1 * beta1 - beta1**2)
    return numerator / denominator * (alpha1 + beta1) ** (lag - 1)


def fourth_standardised_moment(alpha1, beta1):
    """Equation 3, the fourth standardised moment (kurtosis)."""
    return 3.0 + 6.0 * alpha1**2 / _fourth_moment_denominator(alpha1, beta1)


def garch_moment_map(alpha1, beta1, lag: int = 15):
    """The pair :math:`(\\hat\\gamma_n, \\Gamma_4)` the network learns to invert."""
    return normalised_autocovariance(alpha1, beta1, lag), fourth_standardised_moment(
        alpha1, beta1
    )


def empirical_moments(returns, lag: int = 15) -> tuple[float, float]:
    r"""Equations 5-8: the same two statistics, measured on data.

    .. math::
        \hat\gamma_{n} =
          \frac{\overline{(X-\mu)^2(X_{+n}-\mu)^2}
                - \overline{(X-\mu)^2}\;\overline{(X_{+n}-\mu)^2}}
               {\big(\overline{(X-\mu)^2}\big)^2},
        \qquad
        \Gamma_4 = \frac{\overline{(X-\mu)^4}}{\big(\overline{(X-\mu)^2}\big)^2}
    """
    x = np.asarray(returns, dtype=float)
    n = x.size
    if n <= lag + 2:
        raise ValueError(f"Need more than {lag + 2} observations for lag {lag}.")

    centred = x - x.mean()
    squared = centred**2
    left, right = squared[: n - lag], squared[lag:]
    covariance = (left * right).mean() - left.mean() * right.mean()
    gamma = covariance / squared.mean() ** 2
    kurtosis = (centred**4).mean() / squared.mean() ** 2
    return float(gamma), float(kurtosis)


class _MomentInverter(nn.Module):
    """The paper's MLP: ``2 -> 25 -> 205 -> 205 -> 25 -> 1`` (sec. 3.1)."""

    def __init__(self, widths: tuple[int, ...] = (25, 205, 205, 25)):
        super().__init__()
        layers: list[nn.Module] = []
        size = 2
        for width in widths:
            layers += [nn.Linear(size, width, dtype=torch.float64), nn.ReLU()]
            size = width
        layers.append(nn.Linear(size, 1, dtype=torch.float64))
        self.net = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        # alpha_1 lies in (0, 1); a sigmoid enforces that rather than hoping the
        # regression stays in range.
        return torch.sigmoid(self.net(x).squeeze(-1))


class AmortizedGARCHInference:
    r"""Estimate GARCH(1,1) parameters from two moments, via a trained network.

    Parameters
    ----------
    lag
        The ``n`` in :math:`\hat\gamma_n`. The paper uses 15.
    n_samples
        Size of the synthetic training set. The paper uses 125,000, split
        50k/25k/50k into train, validation and test.

    Examples
    --------
    The moment map and its inverse are exact, so a round trip recovers the
    parameters without any network at all — which is the property the network is
    being trained to approximate quickly:

    >>> import torch
    >>> torch.set_default_dtype(torch.float64)
    >>> gamma, kurt = garch_moment_map(0.09, 0.88, lag=15)
    >>> est = AmortizedGARCHInference(lag=15)
    >>> beta = est.beta_from_moments(0.09, kurt)
    >>> round(beta, 6)
    0.88
    """

    provenance = Provenance(
        paper_id="declerk2022_amortized_garch",
        title="AI algorithms for fitting GARCH parameters to empirical financial data",
        authors=("Luke De Clerk", "Sergey Savel'ev"),
        year=2022,
        venue="Physica A 603, 127869",
        doi="10.1016/j.physa.2022.127869",
        equations="Eq. 2 (autocovariance), Eq. 3 (kurtosis), Eq. 4 (variance), Eqs. 5-8",
        code_status=CodeStatus.NO_PUBLIC_CODE_IDENTIFIED,
        community_repos=("https://github.com/jhihan/GARCH_Neural_Network",),
        implementation_status=ImplementationStatus.PAPER_FAITHFUL,
        deviations=(
            "A sigmoid is applied to the network output so alpha_1 lies in (0,1); "
            "the paper does not state an output activation, and an unconstrained "
            "regression can return an inadmissible alpha_1.",
            "beta_1 is recovered by solving Eq. 3 in closed form. The paper says "
            "only that beta_1 'can be evaluated from' it; the positive root is "
            "taken, being the only one with beta_1 >= 0.",
        ),
        notes=(
            "The training set is synthetic by construction: it is the analytic "
            "moment map of Eqs. 2-3, not a stand-in for data. Evaluation is on "
            "real assets."
        ),
    )

    def __init__(self, lag: int = 15, n_samples: int = 125_000, seed: int | None = 0):
        self.lag = lag
        self.n_samples = n_samples
        self.seed = seed
        if seed is not None:
            set_seed(seed)
        self.network = _MomentInverter()
        self._feature_mean = torch.zeros(2, dtype=torch.float64)
        self._feature_scale = torch.ones(2, dtype=torch.float64)
        self.history_: dict[str, list[float]] = {}

    # -- feasibility -------------------------------------------------------- #
    def is_feasible(self, gamma: float, kurtosis: float) -> bool:
        r"""Whether a moment pair is reachable by some GARCH-normal(1,1).

        The paper's Figure 1. A pair outside this region cannot be matched by
        *any* parameter choice, so returning an estimate for it would be
        fabrication. Empirically these are short windows, where too few rare
        events have been seen for the fourth moment to be informative.
        """
        if not np.isfinite(gamma) or not np.isfinite(kurtosis):
            return False
        if kurtosis <= 3.0 or gamma <= 0.0:
            return False   # a GARCH always has excess kurtosis and positive gamma
        alpha1 = self._alpha_from_moments_bruteforce(gamma, kurtosis)
        return alpha1 is not None

    def beta_from_moments(self, alpha1: float, kurtosis: float) -> float:
        r"""Solve Eq. 3 for :math:`\beta_1` given :math:`\alpha_1`.

        Rearranging :math:`\Gamma_4 = 3 + 6\alpha_1^2/D` gives
        :math:`D = 6\alpha_1^2/(\Gamma_4-3)`, and substituting the definition of
        :math:`D` yields a quadratic

        .. math::
            \beta_1^2 + 2\alpha_1\beta_1 + \big(3\alpha_1^2 - 1 + D\big) = 0,
            \qquad
            \beta_1 = -\alpha_1 + \sqrt{1 - 2\alpha_1^2 - D}.

        The positive root is the admissible one.
        """
        if kurtosis <= 3.0:
            raise ValueError("Gamma_4 must exceed 3 for a GARCH with excess kurtosis.")
        d = 6.0 * alpha1**2 / (kurtosis - 3.0)
        discriminant = 1.0 - 2.0 * alpha1**2 - d
        if discriminant < 0.0:
            raise ValueError(
                f"No admissible beta_1 for alpha_1={alpha1:.4f}, Gamma_4={kurtosis:.4f}: "
                "the moment pair is outside the GARCH-feasible region."
            )
        return float(-alpha1 + np.sqrt(discriminant))

    def _alpha_from_moments_bruteforce(
        self, gamma: float, kurtosis: float, grid: int = 2000
    ) -> float | None:
        """Reference inverse by grid search — the target the network approximates.

        Vectorised over the grid. The scalar version spent 2000 Python-level
        calls per query, which made :meth:`estimate` slower than the maximum
        likelihood fit it is supposed to replace — self-defeating for a method
        whose entire selling point is speed.
        """
        if kurtosis <= 3.0:
            return None
        alphas = np.linspace(1e-4, 0.5, grid)
        d = 6.0 * alphas**2 / (kurtosis - 3.0)
        discriminant = 1.0 - 2.0 * alphas**2 - d

        admissible = discriminant >= 0.0
        betas = np.full_like(alphas, np.nan)
        betas[admissible] = -alphas[admissible] + np.sqrt(discriminant[admissible])
        admissible &= (betas >= 0.0) & (alphas + betas < 1.0)
        if not admissible.any():
            return None

        predicted = normalised_autocovariance(alphas[admissible], betas[admissible], self.lag)
        errors = np.abs(predicted - gamma)
        return float(alphas[admissible][int(np.argmin(errors))])

    # -- training ----------------------------------------------------------- #
    def make_training_set(self) -> tuple[Tensor, Tensor]:
        r"""Synthesise :math:`(\hat\gamma_n, \Gamma_4) \to \alpha_1` pairs.

        Draws are rejected unless the fourth moment exists and the process is
        stationary, so every training row is a pair the map can actually reach.
        """
        rng = np.random.default_rng(self.seed)
        features, targets = [], []
        while len(targets) < self.n_samples:
            batch = self.n_samples
            alpha1 = rng.uniform(0.01, 0.35, batch)
            beta1 = rng.uniform(0.30, 0.98, batch)
            denominator = _fourth_moment_denominator(alpha1, beta1)
            ok = (denominator > 1e-4) & (alpha1 + beta1 < 0.999)
            alpha1, beta1 = alpha1[ok], beta1[ok]
            gamma, kurtosis = garch_moment_map(alpha1, beta1, self.lag)
            finite = np.isfinite(gamma) & np.isfinite(kurtosis) & (kurtosis < 1e4)
            features.append(np.column_stack([gamma[finite], kurtosis[finite]]))
            targets.append(alpha1[finite])

        x = np.concatenate(features)[: self.n_samples]
        y = np.concatenate(targets)[: self.n_samples]
        return torch.tensor(x), torch.tensor(y)

    def fit(self, train: TrainConfig | None = None, batch_size: int = 512) -> dict:
        """Train the inverter on synthetic moment pairs (paper sec. 3.1)."""
        cfg = train or TrainConfig(optimizer="adam", lr=1e-3, max_epochs=1000, patience=30)
        x, y = self.make_training_set()

        # Standardise: Gamma_4 can reach hundreds while gamma is order 0.1, and
        # an unstandardised MLP simply ignores the smaller input.
        self._feature_mean = x.mean(dim=0)
        self._feature_scale = x.std(dim=0).clamp(min=1e-8)
        x = (x - self._feature_mean) / self._feature_scale

        n = x.shape[0]
        n_train, n_validation = int(0.4 * n), int(0.2 * n)   # the paper's 50k/25k/50k
        x_train, y_train = x[:n_train], y[:n_train]
        x_validation, y_validation = (
            x[n_train : n_train + n_validation],
            y[n_train : n_train + n_validation],
        )

        optimiser = torch.optim.Adam(self.network.parameters(), lr=cfg.lr)
        history = {"loss": [], "val_loss": []}
        best, best_state, stalled = float("inf"), None, 0

        for _ in range(cfg.max_epochs):
            self.network.train()
            order = torch.randperm(n_train)
            total = 0.0
            for start in range(0, n_train, batch_size):
                index = order[start : start + batch_size]
                optimiser.zero_grad(set_to_none=True)
                loss = torch.nn.functional.mse_loss(self.network(x_train[index]), y_train[index])
                loss.backward()
                optimiser.step()
                total += float(loss.detach()) * index.numel()

            self.network.eval()
            with torch.no_grad():
                validation = float(
                    torch.nn.functional.mse_loss(self.network(x_validation), y_validation)
                )
            history["loss"].append(total / n_train)
            history["val_loss"].append(validation)

            if validation < best - 1e-12:
                best, stalled = validation, 0
                best_state = {k: v.clone() for k, v in self.network.state_dict().items()}
            else:
                stalled += 1
                if stalled >= cfg.patience:
                    break

        if best_state is not None:
            self.network.load_state_dict(best_state)
        self.history_ = history
        return history

    # -- inference ---------------------------------------------------------- #
    def estimate(self, returns, strict: bool = True) -> GARCHParameters:
        r"""Recover :math:`(\alpha_0, \alpha_1, \beta_1)` from a return series.

        Parameters
        ----------
        strict
            When ``True`` (default), a moment pair outside the GARCH-feasible
            region raises. The paper discards such windows; returning a number
            for them would be inventing one.
        """
        gamma, kurtosis = empirical_moments(returns, self.lag)
        feasible = self.is_feasible(gamma, kurtosis)
        if not feasible and strict:
            raise ValueError(
                f"Moment pair (gamma={gamma:.4f}, Gamma_4={kurtosis:.4f}) lies outside "
                "the GARCH-feasible region (paper Fig. 1); no GARCH(1,1) can match it. "
                "Pass strict=False to return a best-effort estimate flagged as infeasible."
            )

        self.network.eval()
        with torch.no_grad():
            features = (
                torch.tensor([[gamma, kurtosis]]) - self._feature_mean
            ) / self._feature_scale
            alpha1 = float(self.network(features).squeeze())

        note = ""
        try:
            beta1 = self.beta_from_moments(alpha1, kurtosis)
        except ValueError as exc:
            beta1 = max(0.0, 0.95 - alpha1)
            note = f"beta_1 fell back to a stationary default: {exc}"

        persistence = alpha1 + beta1
        variance = float(np.var(np.asarray(returns, dtype=float)))
        alpha0 = variance * (1.0 - persistence) if persistence < 1.0 else float("nan")

        return GARCHParameters(
            alpha0=alpha0,
            alpha1=alpha1,
            beta1=beta1,
            feasible=feasible,
            persistence=persistence,
            note=note,
        )

    def warm_start(self, returns):
        """A :class:`~garchai.econometric.garch.GARCH` initialised at the estimate.

        The paper's practical pitch: a near-instant estimate is a good starting
        point for a likelihood optimiser, which then needs far fewer iterations.
        """
        from ..econometric import GARCH

        parameters = self.estimate(returns, strict=False)
        return GARCH(
            omega=max(parameters.alpha0, 1e-6),
            alpha=min(max(parameters.alpha1, 1e-4), 0.4),
            beta=min(max(parameters.beta1, 1e-4), 0.95),
        )


#: The blueprint's name for the same object.
NeuralGARCHEstimator = AmortizedGARCHInference
