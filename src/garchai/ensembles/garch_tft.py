r"""Petrosino et al. (2025): GARCH-Temporal Fusion Transformer for ETF volatility.

A Temporal Fusion Transformer (Lim et al. 2021) whose observed-input block is
augmented with GARCH-family volatility forecasts. The paper's finding is
specific and worth stating precisely, because it is *not* "the hybrid wins":

- On the **Garman-Klass** proxy, GARCH-TFT significantly beats the alternatives.
- On **historical volatility**, it is merely *comparable* to a stand-alone TFT.

So the econometric input helps for one target and not the other. GK uses intraday
high/low/open/close and is the lower-variance proxy; HV is a rolling standard
deviation of closes, which is noisier and already close to what an LSTM extracts
from the return sequence unaided. Reporting only the GK column would misdescribe
the paper, so :meth:`evaluate` computes both when both are supplied.

The TFT machinery is retained rather than simplified, because the paper's
secondary claim is interpretability, which lives entirely in two components:

**Variable selection networks** produce softmax weights over inputs at every time
step, so ``variable_importance()`` reports which inputs the model actually used —
including whether it used the GARCH forecast at all.

**Interpretable multi-head attention** shares the value projection across heads
(Lim et al. Eq. 14-16). Ordinary multi-head attention gives each head its own
values, so the per-head attention maps are not comparable and averaging them is
meaningless; sharing :math:`V` makes the averaged map a genuine statement about
which past time steps mattered.

Reference
---------
Petrosino, L., Bacco, L., Salvati, G., Merone, M., & Papi, M. (2025). A GARCH-
temporal fusion transformer model for the volatility prediction of exchange
traded funds. *Neural Computing and Applications*.
https://doi.org/10.1007/s00521-025-11468-z

Lim, B., Arik, S. O., Loeff, N., & Pfister, T. (2021). Temporal Fusion
Transformers for interpretable multi-horizon time series forecasting.
*International Journal of Forecasting*, 37(4), 1748-1764.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from ..core.config import TrainConfig, set_seed
from ..core.provenance import CodeStatus, ImplementationStatus, Provenance

__all__ = [
    "GARCHTFT",
    "GatedLinearUnit",
    "GatedResidualNetwork",
    "InterpretableMultiHeadAttention",
    "VariableSelectionNetwork",
    "garman_klass",
    "historical_volatility",
    "quantile_loss",
]


# --------------------------------------------------------------------------- #
# volatility proxies (paper sec. 3.2)
# --------------------------------------------------------------------------- #
def historical_volatility(close, window: int = 10) -> np.ndarray:
    r"""Rolling standard deviation of log returns, annualised by :math:`\sqrt{252}`."""
    c = np.asarray(close, dtype=float)
    log_returns = np.diff(np.log(c), prepend=np.log(c[0]))
    out = np.full(c.size, np.nan)
    for t in range(window, c.size):
        out[t] = log_returns[t - window : t].std(ddof=1) * np.sqrt(252.0)
    out[:window] = out[window]
    return out


def garman_klass(high, low, open_, close) -> np.ndarray:
    r"""Garman-Klass (1980) range estimator.

    .. math::
        \sigma^2_{GK} = \tfrac12\big(\ln\tfrac{H}{L}\big)^2
                      - (2\ln 2 - 1)\big(\ln\tfrac{C}{O}\big)^2

    Uses the whole daily range rather than two closing prices, which is why it is
    the lower-variance proxy — and, per the paper, the one the GARCH input helps
    most with.
    """
    h, l, o, c = (np.asarray(v, dtype=float) for v in (high, low, open_, close))
    variance = 0.5 * np.log(h / l) ** 2 - (2 * np.log(2) - 1) * np.log(c / o) ** 2
    return np.sqrt(np.maximum(variance, 0.0) * 252.0)


# --------------------------------------------------------------------------- #
# TFT building blocks (Lim et al. 2021)
# --------------------------------------------------------------------------- #
class GatedLinearUnit(nn.Module):
    r""":math:`\mathrm{GLU}(x) = \sigma(Wx + b) \odot (Vx + c)` — Lim Eq. 5.

    The gate lets the network suppress a whole component when it is not useful,
    which is how a TFT copes with inputs that matter for some series and not
    others without needing a separate architecture per series.
    """

    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(d_model, d_model)
        self.gate = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dropout(x)
        return torch.sigmoid(self.gate(x)) * self.fc(x)


class GatedResidualNetwork(nn.Module):
    """The TFT's universal nonlinear block — Lim Eq. 2-4, with optional context."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int | None = None,
        dropout: float = 0.1,
        context_size: int | None = None,
    ):
        super().__init__()
        output_size = output_size or input_size
        self.skip = (
            nn.Linear(input_size, output_size) if input_size != output_size else nn.Identity()
        )
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.context = (
            nn.Linear(context_size, hidden_size, bias=False) if context_size else None
        )
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.glu = GatedLinearUnit(output_size, dropout)
        self.norm = nn.LayerNorm(output_size)

    def forward(self, x: torch.Tensor, context: torch.Tensor | None = None) -> torch.Tensor:
        hidden = self.fc1(x)
        if self.context is not None and context is not None:
            hidden = hidden + self.context(context)
        hidden = self.fc2(nn.functional.elu(hidden))
        return self.norm(self.skip(x) + self.glu(hidden))


class VariableSelectionNetwork(nn.Module):
    """Softmax weights over inputs at each time step — Lim Eq. 6-8.

    The weights are the model's own statement about which inputs it used, and
    they are what makes ``variable_importance()`` meaningful rather than a
    post-hoc attribution.
    """

    def __init__(
        self,
        n_variables: int,
        d_model: int,
        dropout: float = 0.1,
        context_size: int | None = None,
    ):
        super().__init__()
        self.n_variables = n_variables
        self.flattened_grn = GatedResidualNetwork(
            n_variables * d_model, d_model, n_variables, dropout, context_size
        )
        self.variable_grns = nn.ModuleList(
            GatedResidualNetwork(d_model, d_model, d_model, dropout)
            for _ in range(n_variables)
        )

    def forward(
        self, embeddings: torch.Tensor, context: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """``embeddings`` has shape ``(batch, time, n_variables, d_model)``."""
        batch, time, n_variables, d_model = embeddings.shape
        flat = embeddings.reshape(batch, time, n_variables * d_model)
        weights = torch.softmax(self.flattened_grn(flat, context), dim=-1)
        processed = torch.stack(
            [self.variable_grns[i](embeddings[:, :, i, :]) for i in range(n_variables)],
            dim=2,
        )
        combined = (processed * weights.unsqueeze(-1)).sum(dim=2)
        return combined, weights


class InterpretableMultiHeadAttention(nn.Module):
    """Multi-head attention with a **shared** value projection — Lim Eq. 14-16."""

    def __init__(self, d_model: int, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model={d_model} must be divisible by n_heads={n_heads}.")
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.q = nn.Linear(d_model, d_model)
        self.k = nn.Linear(d_model, d_model)
        self.v = nn.Linear(d_model, self.d_head)   # shared across heads
        self.out = nn.Linear(self.d_head, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, time, _ = q.shape
        query = self.q(q).view(batch, time, self.n_heads, self.d_head).transpose(1, 2)
        key = self.k(k).view(batch, k.shape[1], self.n_heads, self.d_head).transpose(1, 2)
        value = self.v(v).unsqueeze(1)              # one value tensor for all heads

        scores = query @ key.transpose(-2, -1) / (self.d_head**0.5)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attention = self.dropout(torch.softmax(scores, dim=-1))
        head_outputs = attention @ value
        # Averaging heads is legitimate only because they share V.
        return self.out(head_outputs.mean(dim=1)), attention.mean(dim=1)


def quantile_loss(
    prediction: torch.Tensor, target: torch.Tensor, quantiles: tuple[float, ...]
) -> torch.Tensor:
    r"""Pinball loss summed over quantiles — Lim Eq. 22.

    .. math::
        \mathcal{L} = \sum_q \max\big(q(y - \hat{y}_q),\ (q-1)(y - \hat{y}_q)\big)
    """
    losses = []
    for i, q in enumerate(quantiles):
        error = target - prediction[..., i]
        losses.append(torch.max(q * error, (q - 1.0) * error).unsqueeze(-1))
    return torch.cat(losses, dim=-1).sum(dim=-1).mean()


# --------------------------------------------------------------------------- #
# the model
# --------------------------------------------------------------------------- #
class _TFTNetwork(nn.Module):
    """Encoder-decoder TFT over a single sequence window."""

    def __init__(
        self,
        n_variables: int,
        d_model: int = 32,
        n_heads: int = 4,
        dropout: float = 0.1,
        n_quantiles: int = 3,
        n_static: int = 0,
    ):
        super().__init__()
        self.n_variables = n_variables
        self.d_model = d_model
        self.embed = nn.ModuleList(nn.Linear(1, d_model) for _ in range(n_variables))

        self.static_embed = nn.Linear(n_static, d_model) if n_static else None
        context_size = d_model if n_static else None
        if n_static:
            self.static_grns = nn.ModuleList(
                GatedResidualNetwork(d_model, d_model, d_model, dropout) for _ in range(4)
            )

        self.vsn = VariableSelectionNetwork(n_variables, d_model, dropout, context_size)
        self.lstm = nn.LSTM(d_model, d_model, batch_first=True)
        self.post_lstm_gate = GatedLinearUnit(d_model, dropout)
        self.post_lstm_norm = nn.LayerNorm(d_model)
        self.static_enrichment = GatedResidualNetwork(
            d_model, d_model, d_model, dropout, context_size
        )
        self.attention = InterpretableMultiHeadAttention(d_model, n_heads, dropout)
        self.post_attention_gate = GatedLinearUnit(d_model, dropout)
        self.post_attention_norm = nn.LayerNorm(d_model)
        self.position_wise = GatedResidualNetwork(d_model, d_model, d_model, dropout)
        self.output = nn.Linear(d_model, n_quantiles)

    def forward(
        self, x: torch.Tensor, static: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        batch, time, _ = x.shape
        embeddings = torch.stack(
            [self.embed[i](x[:, :, i : i + 1]) for i in range(self.n_variables)], dim=2
        )

        context = enrichment_context = None
        if self.static_embed is not None and static is not None:
            static_vector = self.static_embed(static)
            context = self.static_grns[0](static_vector)
            enrichment_context = self.static_grns[1](static_vector)

        selected, weights = self.vsn(embeddings, context)
        encoded, _ = self.lstm(selected)
        encoded = self.post_lstm_norm(self.post_lstm_gate(encoded) + selected)

        enriched = self.static_enrichment(
            encoded,
            enrichment_context.unsqueeze(1).expand(-1, time, -1)
            if enrichment_context is not None
            else None,
        )
        # Causal mask: position t may attend to <= t only. Without it the model
        # reads the future and every reported metric is meaningless.
        mask = torch.tril(torch.ones(time, time, device=x.device, dtype=x.dtype))
        attended, attention_map = self.attention(enriched, enriched, enriched, mask)
        attended = self.post_attention_norm(self.post_attention_gate(attended) + enriched)

        decoded = self.position_wise(attended)
        # Forecast from the final position of the window.
        return self.output(decoded[:, -1, :]), {
            "variable_weights": weights,
            "attention": attention_map,
        }


class GARCHTFT:
    r"""GARCH-TFT: a Temporal Fusion Transformer with GARCH volatility inputs.

    Parameters
    ----------
    quantiles
        Output quantiles. The median (0.5) is the point forecast; the outer pair
        gives prediction intervals, which is the reason to use the pinball loss
        rather than MSE.
    use_garch
        ``False`` drops the GARCH columns, giving the stand-alone TFT the paper
        compares against. This is the ablation its central claim rests on.

    Examples
    --------
    >>> from garchai.ensembles import GARCHTFT
    >>> model = GARCHTFT(use_garch=False)     # the stand-alone TFT baseline
    >>> model.use_garch
    False
    """

    provenance = Provenance(
        paper_id="petrosino2025_garch_tft",
        title=(
            "A GARCH-temporal fusion transformer model for the volatility prediction "
            "of exchange traded funds"
        ),
        authors=(
            "Lorenzo Petrosino", "Luca Bacco", "Giuliano Salvati",
            "Mario Merone", "Marco Papi",
        ),
        year=2025,
        paper_url="https://doi.org/10.1007/s00521-025-11468-z",
        equations=(
            "TFT blocks per Lim et al. (2021) Eq. 2-8 (GRN/GLU/VSN), Eq. 14-16 "
            "(interpretable attention), Eq. 22 (quantile loss); GARCH augmentation "
            "and the HV/GK proxies per Petrosino et al. sec. 3"
        ),
        code_status=CodeStatus.PAPER_LINKED,
        implementation_status=ImplementationStatus.REPRODUCTION,
        deviations=(
            "The authors' repository (github.com/lbacco/garch-tft) carries no "
            "licence file, so no code was ported from it. The architecture is "
            "implemented from the paper and from Lim et al. (2021), which the "
            "paper cites as its base model.",
            "Single-horizon forecasting from the last encoder position. The paper "
            "reports one-step-ahead results; the multi-horizon decoder of the "
            "original TFT is not exercised.",
            "Static covariates are supported but optional; the paper's ETF sector "
            "labels are not distributed with the data.",
        ),
        data_substitutions=(
            "The paper uses SPY and other US ETFs with intraday OHLC. The worked "
            "example uses public OHLC data for a comparable ETF; Garman-Klass "
            "requires the full daily range, which close-only sources cannot supply.",
        ),
    )

    def __init__(
        self,
        window: int = 20,
        d_model: int = 32,
        n_heads: int = 4,
        dropout: float = 0.1,
        quantiles: tuple[float, ...] = (0.1, 0.5, 0.9),
        use_garch: bool = True,
        seed: int = 0,
    ):
        if 0.5 not in quantiles:
            raise ValueError("quantiles must include the median 0.5 for point forecasts.")
        self.window = window
        self.d_model = d_model
        self.n_heads = n_heads
        self.dropout = dropout
        self.quantiles = quantiles
        self.use_garch = use_garch
        self.seed = seed
        self.network: _TFTNetwork | None = None
        self.names: tuple[str, ...] = ()
        self._mean = self._scale = None

    # -- data plumbing ------------------------------------------------------ #
    def _standardise(self, x: np.ndarray, fit: bool = False) -> np.ndarray:
        if fit:
            self._mean = x.mean(axis=0)
            self._scale = np.where(x.std(axis=0) > 1e-12, x.std(axis=0), 1.0)
        return (x - self._mean) / self._scale

    def _reshape(self, x: np.ndarray) -> torch.Tensor:
        """Left-padded rolling windows: one window per observation."""
        padded = np.vstack([np.repeat(x[:1], self.window - 1, axis=0), x])
        windows = np.stack(
            [padded[t : t + self.window] for t in range(x.shape[0])]
        )
        return torch.as_tensor(windows, dtype=torch.get_default_dtype())

    def fit(
        self,
        features,
        target,
        *,
        train: TrainConfig | None = None,
        static=None,
    ) -> GARCHTFT:
        """Fit on a feature matrix and a volatility-proxy target."""
        train = train or TrainConfig(lr=1e-3, max_epochs=200, patience=20)
        set_seed(self.seed)

        x = np.asarray(features, dtype=float)
        y = np.asarray(target, dtype=float)
        if x.shape[0] != y.shape[0]:
            raise ValueError("features and target must align in length.")

        windows = self._reshape(self._standardise(x, fit=True))
        dtype = torch.get_default_dtype()
        y_tensor = torch.as_tensor(y, dtype=dtype)
        static_tensor = (
            torch.as_tensor(np.asarray(static, dtype=float), dtype=dtype)
            if static is not None
            else None
        )

        self.network = _TFTNetwork(
            x.shape[1], self.d_model, self.n_heads, self.dropout,
            len(self.quantiles), static_tensor.shape[1] if static_tensor is not None else 0,
        )
        optimiser = torch.optim.Adam(self.network.parameters(), lr=train.lr)

        history: dict[str, list[float]] = {"loss": []}
        best, patience, best_state = float("inf"), 0, None
        for _ in range(train.max_epochs):
            self.network.train()
            optimiser.zero_grad()
            prediction, _ = self.network(windows, static_tensor)
            loss = quantile_loss(prediction, y_tensor, self.quantiles)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 1.0)
            optimiser.step()

            value = float(loss.detach())
            history["loss"].append(value)
            if value < best - 1e-10:
                best, patience = value, 0
                best_state = {k: v.clone() for k, v in self.network.state_dict().items()}
            else:
                patience += 1
                if patience >= train.patience:
                    break
        if best_state is not None:
            self.network.load_state_dict(best_state)
        self.history = history
        return self

    def predict(self, features, static=None) -> np.ndarray:
        """Median (0.5) forecasts."""
        return self.predict_quantiles(features, static)[:, self.quantiles.index(0.5)]

    def predict_quantiles(self, features, static=None) -> np.ndarray:
        """All quantiles, shape ``(T, n_quantiles)`` — the interval forecast."""
        if self.network is None:
            raise RuntimeError("Model has not been fitted; call .fit(...) first.")
        self.network.eval()
        windows = self._reshape(self._standardise(np.asarray(features, dtype=float)))
        static_tensor = (
            torch.as_tensor(np.asarray(static, dtype=float)) if static is not None else None
        )
        with torch.no_grad():
            prediction, _ = self.network(windows, static_tensor)
        return prediction.numpy()

    def quantile_crossing_rate(self, features, static=None) -> float:
        r"""Fraction of predicted quantile pairs that are out of order.

        Nothing constrains :math:`\hat{y}_{0.1} \le \hat{y}_{0.9}`: each quantile
        gets its own output unit and the pinball loss is separable across them,
        so they are free to cross. When they do, the "interval" is empty and the
        coverage number computed from it is not a coverage number.

        The original TFT has this property too and does not enforce monotonicity,
        so we do not silently sort the outputs — that would hide a real symptom
        of an undertrained or overfit model. Check this before quoting
        ``interval_coverage``; a rate above a few percent means the intervals
        should not be reported.
        """
        q = self.predict_quantiles(features, static)
        if q.shape[1] < 2:
            return 0.0
        return float((np.diff(q, axis=1) < 0).mean())

    def variable_importance(self, features, static=None) -> dict[str, float]:
        """Mean variable-selection weight per input — the interpretability claim.

        If the GARCH columns receive negligible weight, the hybrid is not
        actually using its econometric input, whatever its error metrics say.
        """
        if self.network is None:
            raise RuntimeError("Model has not been fitted; call .fit(...) first.")
        self.network.eval()
        windows = self._reshape(self._standardise(np.asarray(features, dtype=float)))
        static_tensor = (
            torch.as_tensor(np.asarray(static, dtype=float)) if static is not None else None
        )
        with torch.no_grad():
            _, interpretation = self.network(windows, static_tensor)
        weights = interpretation["variable_weights"].mean(dim=(0, 1)).numpy()
        names = self.names or tuple(f"x{i}" for i in range(weights.size))
        return dict(zip(names, (float(w) for w in weights)))

    def attention_map(self, features, static=None) -> np.ndarray:
        """Averaged attention over the window — Lim's interpretable map."""
        if self.network is None:
            raise RuntimeError("Model has not been fitted; call .fit(...) first.")
        self.network.eval()
        windows = self._reshape(self._standardise(np.asarray(features, dtype=float)))
        static_tensor = (
            torch.as_tensor(np.asarray(static, dtype=float)) if static is not None else None
        )
        with torch.no_grad():
            _, interpretation = self.network(windows, static_tensor)
        return interpretation["attention"].mean(dim=0).numpy()

    def evaluate(self, features, target, static=None) -> dict[str, float]:
        prediction = self.predict(features, static)
        truth = np.asarray(target, dtype=float)
        residual = truth - prediction
        metrics = {
            "MAE": float(np.mean(np.abs(residual))),
            "RMSE": float(np.sqrt(np.mean(residual**2))),
            "R2": float(1.0 - residual.var() / truth.var()),
        }
        # Interval coverage — the reason for the pinball loss. A 0.1/0.9 pair
        # should contain about 80% of outcomes; reporting only the point error
        # would leave this unchecked. Coverage measured in-sample is not
        # evidence: an overfit model narrows its own band around the values it
        # has memorised, so this must be read on held-out data.
        quantile_predictions = self.predict_quantiles(features, static)
        if len(self.quantiles) >= 2:
            lower, upper = quantile_predictions[:, 0], quantile_predictions[:, -1]
            nominal = self.quantiles[-1] - self.quantiles[0]
            metrics["interval_coverage"] = float(((truth >= lower) & (truth <= upper)).mean())
            metrics["nominal_coverage"] = float(nominal)
            # Crossed quantiles make the coverage figure above meaningless, so
            # it travels with the number rather than being left to be asked for.
            metrics["quantile_crossing_rate"] = float(
                (np.diff(quantile_predictions, axis=1) < 0).mean()
            )
        return metrics
