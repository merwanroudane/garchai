r"""Xu et al. (2024): GARCH-Informed Neural Networks.

The third way of combining GARCH with a network. Zhao puts GARCH in the
architecture, Wei puts it in the gating — GINN puts it in the **loss**:

.. math::
    \mathcal{L} = \lambda\,\mathrm{MSE}\big(\sigma^2_t,\;\hat\sigma^2_{t,GINN}\big)
        + (1-\lambda)\,\mathrm{MSE}\big(\hat\sigma^2_{t,GARCH},\;
          \hat\sigma^2_{t,GINN}\big)
    \qquad\text{(Eq. 17)}

The network is pulled toward the observed variance *and* toward a GARCH model's
forecast, exactly as a physics-informed network is pulled toward data and toward
a governing equation. GARCH supplies the prior where the data is uninformative.

The tuned weight is worth pausing over. Xu et al. find :math:`\lambda = 0.01`
optimal — **99% of the weight on agreeing with GARCH**, 1% on the observed
variance. And GINN-0, the :math:`\lambda = 0` limit that trains purely to imitate
GARCH, performs comparably. That is not a claim that data is unimportant; it is a
statement about how noisy :math:`(r_t - \hat\mu_t)^2` is as a single-observation
target. The GARCH forecast is a smoothed, lower-variance version of the same
signal, and at a 90-day window it carries more usable information than the raw
squared residual does.

Pipeline (Eqs. 13-16)
---------------------
1. An AR model over a rolling 90-day window gives :math:`\hat\mu_t`.
2. The "ground truth" is :math:`\sigma^2_t = (r_t - \hat\mu_t)^2` — Eq. 15.
3. A constant-mean GARCH(1,1) over the same window gives
   :math:`\hat\sigma^2_{t,GARCH}`.
4. The network maps :math:`(\sigma^2_{t-90},\dots,\sigma^2_{t-1})` to
   :math:`\hat\sigma^2_{t,GINN}` — Eq. 16.

Note what the network consumes: **past ground-truth variances**, not returns.
That keeps it strictly a variance-to-variance map, and keeps the comparison with
GARCH like for like.

Reference
---------
Xu, Z., Liechty, J., Benthall, S., Skar-Gislinge, N., & McComb, C. (2024).
GARCH-Informed Neural Networks for Volatility Prediction in Financial Markets.
https://arxiv.org/abs/2410.00288
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

from ..core.config import TrainConfig, set_seed
from ..core.provenance import CodeStatus, ImplementationStatus, Provenance

__all__ = ["GINNDataset", "GARCHInformedNetwork", "GINN", "GINN0", "build_ginn_inputs"]

_XU = dict(
    paper_id="xu2024_ginn",
    title="GARCH-Informed Neural Networks for Volatility Prediction in Financial Markets",
    authors=(
        "Zeda Xu", "John Liechty", "Sebastian Benthall",
        "Nicholas Skar-Gislinge", "Christopher McComb",
    ),
    year=2024,
    paper_url="https://arxiv.org/abs/2410.00288",
    code_status=CodeStatus.COMMUNITY_REPLICATION,
    community_repos=(
        "https://github.com/yeasinstat/GarchInformedLSTM",
        "https://github.com/datafox-tw/GLoVE",
    ),
)


@dataclass
class GINNDataset:
    """The three aligned series GINN needs, plus the window that produced them."""

    targets: np.ndarray
    """Ground-truth variance, :math:`(r_t - \\hat\\mu_t)^2` (Eq. 15)."""

    garch_forecast: np.ndarray
    """Rolling GARCH(1,1) variance forecast for the same periods."""

    features: np.ndarray
    """Shape ``(n, window)``: the lagged ground-truth variances of Eq. 16."""

    window: int = 90

    def __len__(self) -> int:
        return len(self.targets)

    def split(self, train_fraction: float = 0.7) -> tuple[GINNDataset, GINNDataset]:
        """Chronological split. Never random — that would leak the future."""
        cut = int(round(train_fraction * len(self)))
        return (
            GINNDataset(self.targets[:cut], self.garch_forecast[:cut],
                        self.features[:cut], self.window),
            GINNDataset(self.targets[cut:], self.garch_forecast[cut:],
                        self.features[cut:], self.window),
        )


def build_ginn_inputs(
    returns: np.ndarray,
    window: int = 90,
    mean_lags: int = 1,
    refit_garch_every: int = 1,
    verbose: bool = False,
) -> GINNDataset:
    r"""Run the paper's rolling pipeline (Eqs. 13-16).

    For each period ``t``, an AR mean and a GARCH(1,1) are estimated on the
    preceding ``window`` observations only — so every number entering the
    network was computable at the time. Refitting a GARCH at every one of
    several thousand steps is the expensive part; ``refit_garch_every`` reuses
    parameters for a few steps at a time when speed matters, and the deviation
    is recorded on the model.

    Parameters
    ----------
    returns
        Daily returns, in percent.
    window
        Rolling window. The paper uses 90.
    mean_lags
        AR order for :math:`\hat\mu_t`. The paper says "an AR model" without
        fixing the order; 1 is the default.

    Returns
    -------
    GINNDataset
        Aligned targets, GARCH forecasts and lagged-variance features.
    """
    try:
        from arch import arch_model
    except ImportError as exc:  # pragma: no cover
        raise ImportError("build_ginn_inputs needs `arch`: pip install garchai[benchmarks]") from exc

    r = np.asarray(returns, dtype=float)
    n = r.size
    if n <= 2 * window:
        raise ValueError(f"Need more than {2 * window} observations for a {window}-day window.")

    truth = np.full(n, np.nan)
    garch = np.full(n, np.nan)
    cached: dict | None = None

    for t in range(window, n):
        past = r[t - window : t]
        # Eq. 13: AR mean over the trailing window.
        design = np.column_stack(
            [np.ones(window - mean_lags)]
            + [past[mean_lags - k - 1 : window - k - 1] for k in range(mean_lags)]
        )
        coefficients, *_ = np.linalg.lstsq(design, past[mean_lags:], rcond=None)
        mu = coefficients[0] + sum(
            coefficients[k + 1] * r[t - k - 1] for k in range(mean_lags)
        )
        truth[t] = (r[t] - mu) ** 2                              # Eq. 15

        # Eq. 14: GARCH(1,1) forecast from the same window.
        if cached is None or (t - window) % refit_garch_every == 0:
            try:
                fitted = arch_model(past, mean="Constant", vol="GARCH", p=1, q=1).fit(disp="off")
                cached = {"params": fitted.params}
            except Exception:  # pragma: no cover - occasional non-convergence
                cached = cached or {"params": None}
        if cached["params"] is not None:
            p = cached["params"]
            omega, alpha, beta = p["omega"], p["alpha[1]"], p["beta[1]"]
            resid = past - p["mu"]
            h = resid.var()
            for value in resid:
                h = omega + alpha * value**2 + beta * h
            garch[t] = h
        if verbose and t % 500 == 0:  # pragma: no cover
            print(f"  window {t}/{n}")

    # Eq. 16: the network sees the last `window` ground-truth variances.
    valid = np.arange(2 * window, n)
    features = np.stack([truth[t - window : t] for t in valid])
    keep = np.isfinite(features).all(axis=1) & np.isfinite(truth[valid]) & np.isfinite(garch[valid])
    return GINNDataset(truth[valid][keep], garch[valid][keep], features[keep], window)


class GARCHInformedNetwork(nn.Module):
    """The paper's architecture: 3 LSTM layers of width 256, then a dense head.

    Reproduced as specified (sec. II-D): dropout between LSTM layers, two linear
    layers, a ``BatchNorm`` and a ``ReLU``. The final ``softplus`` is ours — the
    paper does not state an output activation, but a variance forecast that can
    go negative is not a variance, and MSE against a positive target would
    happily produce one.
    """

    def __init__(self, window: int = 90, hidden: int = 256, layers: int = 3,
                 dropout: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=1, hidden_size=hidden, num_layers=layers,
            batch_first=True, dropout=dropout, dtype=torch.float64,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden // 2, dtype=torch.float64),
            nn.BatchNorm1d(hidden // 2, dtype=torch.float64),
            nn.ReLU(),
            nn.Linear(hidden // 2, 1, dtype=torch.float64),
        )

    def forward(self, x: Tensor) -> Tensor:
        out, _ = self.lstm(x.unsqueeze(-1))
        return nn.functional.softplus(self.head(out[:, -1, :]).squeeze(-1))


class GINN:
    r"""GARCH-Informed Neural Network (Xu et al., 2024).

    Parameters
    ----------
    lam
        Weight on the data term in Eq. 17. The paper's parametric study over
        NASDAQ selects ``0.01``. ``lam=0`` is :class:`GINN0`.
    window
        Lookback. The paper uses 90 days.
    standardize
        Scale inputs and targets by the training-set mean before fitting.
        Variances of daily percent returns are ~1, but on other scales an
        unstandardised MSE makes the two loss terms incomparable. Statistics
        come from the **training set only**.

    Examples
    --------
    >>> import numpy as np
    >>> from garchai.informed import GINN
    >>> model = GINN(lam=0.01, window=5)
    >>> model.lam
    0.01
    """

    provenance = Provenance(
        **_XU,
        equations="Eqs. 13-16 (pipeline), Eq. 17 (informed loss), Eq. 18 (GINN-0)",
        implementation_status=ImplementationStatus.REPRODUCTION,
        deviations=(
            "The paper states no output activation; a softplus is applied so the "
            "variance forecast cannot go negative.",
            "The AR order for mu_hat is not specified in the paper; it defaults "
            "to 1 and is exposed as mean_lags.",
            "refit_garch_every lets the rolling GARCH be refitted less often "
            "than every step, for speed. The default of 1 is the paper's.",
        ),
    )

    def __init__(
        self,
        lam: float = 0.01,
        window: int = 90,
        hidden: int = 256,
        layers: int = 3,
        dropout: float = 0.2,
        standardize: bool = True,
        seed: int | None = 0,
    ):
        if not 0.0 <= lam <= 1.0:
            raise ValueError("lam must lie in [0, 1].")
        self.lam = lam
        self.window = window
        self.standardize = standardize
        self.seed = seed
        if seed is not None:
            set_seed(seed)
        self.network = GARCHInformedNetwork(window, hidden, layers, dropout)
        self._scale = 1.0
        self.history_: dict[str, list[float]] = {}

    def _prepare(self, data: GINNDataset, fit_scale: bool) -> tuple[Tensor, Tensor, Tensor]:
        if fit_scale and self.standardize:
            self._scale = float(np.mean(data.targets)) or 1.0
        s = self._scale
        return (
            torch.tensor(data.features / s),
            torch.tensor(data.targets / s),
            torch.tensor(data.garch_forecast / s),
        )

    def loss(self, prediction: Tensor, truth: Tensor, garch: Tensor) -> Tensor:
        """Equation 17."""
        from ..losses import garch_informed_loss

        return garch_informed_loss(prediction, truth, garch, lam=self.lam)

    def fit(self, data: GINNDataset, train: TrainConfig | None = None,
            batch_size: int = 128) -> dict:
        """Train on a :class:`GINNDataset`. The paper uses AdamW for 300 epochs."""
        cfg = train or TrainConfig(optimizer="adamw", lr=1e-3, max_epochs=300, patience=30)
        x, y, g = self._prepare(data, fit_scale=True)

        optimiser = torch.optim.AdamW(self.network.parameters(), lr=cfg.lr)
        n = x.shape[0]
        history = {"loss": []}
        best, best_state, stalled = float("inf"), None, 0

        for epoch in range(cfg.max_epochs):
            self.network.train()
            # Contiguous mini-batches; the window itself carries the ordering,
            # so shuffling batch *starts* is safe where shuffling time is not.
            order = torch.randperm(n)
            epoch_loss = 0.0
            for start in range(0, n, batch_size):
                index = order[start : start + batch_size]
                if index.numel() < 2:
                    continue  # BatchNorm needs at least two rows
                optimiser.zero_grad(set_to_none=True)
                prediction = self.network(x[index])
                value = self.loss(prediction, y[index], g[index])
                value.backward()
                optimiser.step()
                epoch_loss += float(value.detach()) * index.numel()

            epoch_loss /= n
            history["loss"].append(epoch_loss)
            if epoch_loss < best - 1e-12:
                best, stalled = epoch_loss, 0
                best_state = {k: v.detach().clone() for k, v in self.network.state_dict().items()}
            else:
                stalled += 1
                if stalled >= cfg.patience:
                    break

        if best_state is not None:
            self.network.load_state_dict(best_state)
        self.history_ = history
        return history

    def predict(self, data: GINNDataset) -> np.ndarray:
        """Variance forecasts on the original scale."""
        self.network.eval()
        x, _, _ = self._prepare(data, fit_scale=False)
        with torch.no_grad():
            return (self.network(x) * self._scale).numpy()

    def evaluate(self, data: GINNDataset) -> dict[str, float]:
        """R-squared, MSE and MAE against the ground-truth variance (paper sec. II-G)."""
        prediction = self.predict(data)
        truth = data.targets
        residual = truth - prediction
        return {
            "R2": float(1.0 - residual.var() / truth.var()),
            "MSE": float(np.mean(residual**2)),
            "MAE": float(np.mean(np.abs(residual))),
        }


class GINN0(GINN):
    r"""GINN-0: the :math:`\lambda = 0` limit (Eq. 18).

    Trains purely to imitate the GARCH forecast, ignoring the observed variance
    entirely. The paper reports it performing comparably to full GINN, which is
    the most interesting result in the paper: the rolling GARCH forecast alone
    carries nearly all the usable signal, and the raw squared residual adds
    almost nothing beyond noise.
    """

    def __init__(self, **kwargs):
        kwargs["lam"] = 0.0
        super().__init__(**kwargs)
