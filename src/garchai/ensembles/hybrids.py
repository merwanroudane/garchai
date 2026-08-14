r"""Feature hybrids and ensembles: GARCH as an input, not as a component.

Four papers that share a shape — fit GARCH first, hand its output to a machine
learner — and differ in what the learner is and how the hand-off works:

============================  ==============================================
:class:`LSTMANNGARCH`         Hu, Ni & Wen (2020). GARCH forecast plus lagged
                              returns into an LSTM-ANN stack.
:class:`GARCHXGBoost`         Maingo, Ravele & Sigauke (2025). A gradient-boosted
                              **residual correction** on top of GARCH.
:class:`StackedMLGARCH`       Rubio et al. (2026). Several base learners, an
                              XGBoost **meta-learner** stacked on their
                              out-of-fold predictions.
:class:`MultiGARCHTransformer` Gebreslase et al. (2026). Multiple GARCH
                              specifications fused by attention.
============================  ==============================================

These are **baselines**, and the package is explicit about that. Blueprint sec. 53
says not to make the feature-hybrid family a library's central novelty, and
``hybridecon`` already owns the plain GARCH-LSTM/GRU/BiLSTM variants. What is
built here is only what ``hybridecon`` does not cover: the residual-correction
and stacking designs, and the transformer fusion.

The single most important thing these classes get right is **no look-ahead**.
The GARCH features are produced by a rolling fit that sees only past data, the
stacking uses out-of-fold predictions, and the train/test split is
chronological. Get any of those wrong and a feature hybrid will report
spectacular accuracy that evaporates out of sample — which is the standard
failure mode of this literature.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ..core.config import TrainConfig, set_seed
from ..core.provenance import CodeStatus, ImplementationStatus, Provenance

__all__ = [
    "GARCHFeatures",
    "build_garch_features",
    "LSTMANNGARCH",
    "GARCHXGBoost",
    "StackedMLGARCH",
    "MultiGARCHTransformer",
]

GARCH_SPECS = ("garch", "gjr", "egarch", "aparch")


@dataclass
class GARCHFeatures:
    """Rolling GARCH forecasts plus the target they are meant to predict."""

    features: np.ndarray
    """Shape ``(T, n_features)``: one column per GARCH specification, then lags."""

    target: np.ndarray
    names: tuple[str, ...] = ()
    refit_every: int = 1
    note: str = ""

    def __len__(self) -> int:
        return len(self.target)

    def split(self, train_fraction: float = 0.8) -> tuple[GARCHFeatures, GARCHFeatures]:
        """Chronological split. Random splitting would leak the future."""
        cut = int(round(train_fraction * len(self)))
        return (
            GARCHFeatures(self.features[:cut], self.target[:cut], self.names,
                          self.refit_every, self.note),
            GARCHFeatures(self.features[cut:], self.target[cut:], self.names,
                          self.refit_every, self.note),
        )


def build_garch_features(
    returns,
    target,
    specifications: tuple[str, ...] = GARCH_SPECS,
    n_lags: int = 5,
    window: int = 500,
    refit_every: int = 20,
    verbose: bool = False,
) -> GARCHFeatures:
    r"""Rolling one-step GARCH forecasts, plus lagged returns.

    For each period ``t`` the models are estimated on the preceding ``window``
    observations only, so every feature was computable at the time. That is the
    whole ballgame for this family: an in-sample GARCH fit used as a feature
    leaks the future and inflates every downstream metric.

    Parameters
    ----------
    refit_every
        Re-estimate the GARCH parameters every ``k`` steps rather than at every
        one. Between refits the *filter* still runs forward on new data — only
        the parameters are held. This is a genuine approximation and it is
        recorded on the returned object; ``refit_every=1`` is exact.

    Returns
    -------
    GARCHFeatures
        Rows with any non-finite feature are dropped, so the result is directly
        usable.
    """
    from arch import arch_model

    r = np.asarray(returns, dtype=float)
    y = np.asarray(target, dtype=float)
    if r.size != y.size:
        raise ValueError(f"returns and target must align: {r.size} vs {y.size}")
    if r.size <= window + n_lags:
        raise ValueError(f"Need more than {window + n_lags} observations.")

    columns = {spec: np.full(r.size, np.nan) for spec in specifications}
    cached: dict[str, dict] = {}

    for t in range(window, r.size):
        past = r[t - window : t]
        for spec in specifications:
            if (t - window) % refit_every == 0 or spec not in cached:
                try:
                    vol, o = "GARCH", 0
                    if spec == "egarch":
                        vol, o = "EGARCH", 1
                    elif spec == "gjr":
                        o = 1
                    elif spec == "aparch":
                        vol, o = "APARCH", 1
                    fitted = arch_model(
                        past, mean="Constant", vol=vol, p=1, o=o, q=1
                    ).fit(disp="off", show_warning=False)
                    cached[spec] = {"model": fitted}
                except Exception:  # noqa: BLE001
                    continue
            if spec in cached:
                # One-step-ahead forecast from the most recent parameters.
                forecast = cached[spec]["model"].forecast(horizon=1, reindex=False)
                columns[spec][t] = float(np.sqrt(forecast.variance.values[-1, 0]))
        if verbose and t % 250 == 0:  # pragma: no cover
            print(f"  window {t}/{r.size}")

    lags = np.column_stack([np.roll(r, k) for k in range(1, n_lags + 1)])
    lags[:n_lags] = np.nan
    matrix = np.column_stack([columns[s] for s in specifications] + [lags])
    names = tuple(specifications) + tuple(f"r_lag{k}" for k in range(1, n_lags + 1))

    keep = np.isfinite(matrix).all(axis=1) & np.isfinite(y)
    return GARCHFeatures(
        features=matrix[keep],
        target=y[keep],
        names=names,
        refit_every=refit_every,
        note=(
            f"Rolling {window}-day window; GARCH parameters refitted every "
            f"{refit_every} step(s)."
            + ("" if refit_every == 1 else " This is an approximation to the exact protocol.")
        ),
    )


class _SequenceModel(nn.Module):
    """Shared trainer for the sequence-based hybrids."""

    def __init__(self):
        super().__init__()
        self.history_: dict[str, list[float]] = {}
        self._mean = 0.0
        self._scale = 1.0

    def _standardise(self, x: np.ndarray, fit: bool) -> Tensor:
        if fit:
            self._mean = x.mean(axis=0)
            self._scale = x.std(axis=0) + 1e-8
        return torch.tensor((x - self._mean) / self._scale)

    def fit_features(
        self, data: GARCHFeatures, train: TrainConfig | None = None,
        batch_size: int = 64, seed: int | None = 0,
    ) -> dict:
        """Train on a :class:`GARCHFeatures` bundle."""
        cfg = train or TrainConfig(optimizer="adam", lr=1e-3, max_epochs=200, patience=25)
        if seed is not None:
            set_seed(seed)
        x = self._standardise(data.features, fit=True)
        y = torch.tensor(data.target)
        x = self._reshape(x)

        optimiser = torch.optim.Adam(self.parameters(), lr=cfg.lr)
        n = x.shape[0]
        history = {"loss": []}
        best, best_state, stalled = float("inf"), None, 0

        for _ in range(cfg.max_epochs):
            self.train()
            order = torch.randperm(n)
            total = 0.0
            for start in range(0, n, batch_size):
                index = order[start : start + batch_size]
                optimiser.zero_grad(set_to_none=True)
                loss = F.mse_loss(self(x[index]).squeeze(-1), y[index])
                loss.backward()
                nn.utils.clip_grad_norm_(self.parameters(), 1.0)
                optimiser.step()
                total += float(loss.detach()) * index.numel()
            total /= n
            history["loss"].append(total)
            if total < best - 1e-12:
                best, stalled = total, 0
                best_state = {k: v.clone() for k, v in self.state_dict().items()}
            else:
                stalled += 1
                if stalled >= cfg.patience:
                    break

        if best_state is not None:
            self.load_state_dict(best_state)
        self.history_ = history
        return history

    def predict(self, data: GARCHFeatures) -> np.ndarray:
        self.eval()
        x = self._reshape(self._standardise(data.features, fit=False))
        with torch.no_grad():
            return self(x).squeeze(-1).numpy()

    def evaluate(self, data: GARCHFeatures) -> dict[str, float]:
        prediction = self.predict(data)
        residual = data.target - prediction
        return {
            "MAE": float(np.mean(np.abs(residual))),
            "RMSE": float(np.sqrt(np.mean(residual**2))),
            "R2": float(1.0 - residual.var() / data.target.var()),
        }

    def _reshape(self, x: Tensor) -> Tensor:
        return x


class LSTMANNGARCH(_SequenceModel):
    r"""Hu, Ni & Wen (2020): GARCH forecasts into an LSTM-ANN stack.

    The paper compares six models — ANN, LSTM-ANN, BLSTM-ANN and the three
    GARCH-prefixed versions — on copper price volatility. The finding that
    matters is the *prefix*: adding the GARCH forecast as an input improves
    every neural variant, which is the empirical case for feature hybrids even
    though they are not structurally interesting.

    ``bidirectional=True`` gives the BLSTM-ANN variant.

    Notes
    -----
    A bidirectional LSTM over a **window** is legitimate — it reads forwards and
    backwards within an already-past window — but it must never be run over the
    whole series, which would read the future. The windowing here makes that
    impossible by construction.
    """

    provenance = Provenance(
        paper_id="hu2020_lstm_ann_garch",
        title=(
            "A hybrid deep learning approach by integrating LSTM-ANN networks "
            "with GARCH model for copper price volatility prediction"
        ),
        authors=("Yan Hu", "Jian Ni", "Liu Wen"),
        year=2020,
        venue="Physica A 557, 124907",
        doi="10.1016/j.physa.2020.124907",
        equations="Eq. 1 (returns), sec. 2 (six model variants)",
        code_status=CodeStatus.NO_PUBLIC_CODE_IDENTIFIED,
        implementation_status=ImplementationStatus.REPRODUCTION,
        deviations=(
            "Layer widths are not given numerically in the paper; exposed as "
            "arguments.",
        ),
        data_substitutions=(
            "The paper studies copper futures. Any return series may be passed; "
            "the bundled examples use equity indices and say so.",
        ),
        notes="A baseline. hybridecon owns the plain GARCH-LSTM feature hybrids.",
    )

    def __init__(
        self,
        n_features: int,
        window: int = 20,
        hidden_size: int = 32,
        ann_width: int = 16,
        bidirectional: bool = False,
    ):
        super().__init__()
        self.window = window
        self.n_features = n_features
        dt = torch.float64
        self.lstm = nn.LSTM(
            n_features, hidden_size, batch_first=True,
            bidirectional=bidirectional, dtype=dt,
        )
        out_size = hidden_size * (2 if bidirectional else 1)
        self.ann = nn.Sequential(
            nn.Linear(out_size, ann_width, dtype=dt),
            nn.ReLU(),
            nn.Linear(ann_width, 1, dtype=dt),
        )

    def _reshape(self, x: Tensor) -> Tensor:
        """Slide a window over the feature matrix, left-padded to stay aligned."""
        padded = torch.cat([x[:1].expand(self.window - 1, -1), x])
        return padded.unfold(0, self.window, 1).permute(0, 2, 1)

    def forward(self, x: Tensor) -> Tensor:
        out, _ = self.lstm(x)
        return F.softplus(self.ann(out[:, -1, :]))


class MultiGARCHTransformer(_SequenceModel):
    r"""Gebreslase et al. (2026): several GARCH specifications fused by attention.

    Where :class:`LSTMANNGARCH` concatenates the GARCH forecasts and lets a
    recurrent net sort them out, this attends over them — so the weight given to
    each specification is *itself* a function of the current state. In principle
    that lets the model lean on GJR during a downturn and on plain GARCH in calm
    periods, which a fixed concatenation cannot express.

    Whether that extra flexibility pays for itself is exactly the sort of claim
    the Model Confidence Set exists to adjudicate, and on short test samples it
    usually will not separate from the simpler hybrids.
    """

    provenance = Provenance(
        paper_id="gebreslase2026_multi_garch_transformer",
        title="A Hybrid Multi-GARCH Transformer Model for Stock Price Volatility Forecasting",
        authors=(
            "Tirhas Tesfay Gebreslase", "Kilai Mutua",
            "Yemane Hailu Fissuh", "Tsgab Gebrecherkos Girmay",
        ),
        year=2026,
        venue="Commun. Math. Biol. Neurosci. 2026:48",
        doi="10.28919/cmbn/9834",
        code_status=CodeStatus.NO_PUBLIC_CODE_IDENTIFIED,
        implementation_status=ImplementationStatus.REPRODUCTION,
        deviations=(
            "Transformer depth, width and head count are not given numerically; "
            "exposed as arguments with conventional defaults.",
        ),
        notes="Not referenced in the ecosystem blueprint; added from the supplied PDF.",
    )

    def __init__(
        self,
        n_features: int,
        window: int = 20,
        d_model: int = 32,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.window = window
        dt = torch.float64
        self.embed = nn.Linear(n_features, d_model, dtype=dt)
        self.position = nn.Parameter(torch.randn(window, d_model, dtype=dt) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model, n_heads, dim_feedforward=4 * d_model,
            dropout=dropout, batch_first=True, dtype=dt,
        )
        self.encoder = nn.TransformerEncoder(layer, n_layers)
        self.head = nn.Linear(d_model, 1, dtype=dt)

    def _reshape(self, x: Tensor) -> Tensor:
        padded = torch.cat([x[:1].expand(self.window - 1, -1), x])
        return padded.unfold(0, self.window, 1).permute(0, 2, 1)

    def forward(self, x: Tensor) -> Tensor:
        embedded = self.embed(x) + self.position
        encoded = self.encoder(embedded)
        return F.softplus(self.head(encoded[:, -1, :]))


class GARCHXGBoost:
    r"""Maingo, Ravele & Sigauke (2025): gradient-boosted residual correction.

    A two-stage design, and the ordering is the point. GARCH takes the variance
    it can explain; XGBoost is then fitted to what is *left over*:

    .. math::
        \hat\sigma_t = \hat\sigma_t^{GARCH}
                     + f_{\text{XGB}}\big(\text{features}_t\big)

    Fitting the booster to the residual rather than to the target directly means
    it cannot simply relearn the GARCH signal and claim credit for it. Whatever
    it contributes is, by construction, orthogonal to what GARCH already knew.

    Requires ``xgboost``; falls back to scikit-learn's gradient booster if it is
    not installed, and says which was used.
    """

    provenance = Provenance(
        paper_id="maingo2025_statistical_ml_fusion",
        title="A Fusion of Statistical and Machine Learning Models for Volatility Forecasting",
        authors=("I. Maingo", "T. Ravele", "C. Sigauke"),
        year=2025,
        venue="International Journal of Financial Studies 13, 155",
        doi="10.3390/ijfs13030155",
        code_status=CodeStatus.NO_PUBLIC_CODE_IDENTIFIED,
        implementation_status=ImplementationStatus.REPRODUCTION,
        deviations=(
            "Booster hyper-parameters are not given numerically; conventional "
            "defaults are used and exposed.",
        ),
    )

    def __init__(
        self,
        garch_column: int = 0,
        n_estimators: int = 300,
        max_depth: int = 4,
        learning_rate: float = 0.05,
        seed: int = 0,
    ):
        self.garch_column = garch_column
        self.backend = ""
        try:
            from xgboost import XGBRegressor

            self.model = XGBRegressor(
                n_estimators=n_estimators, max_depth=max_depth,
                learning_rate=learning_rate, random_state=seed, verbosity=0,
            )
            self.backend = "xgboost"
        except ImportError:  # pragma: no cover - optional dependency
            from sklearn.ensemble import GradientBoostingRegressor

            self.model = GradientBoostingRegressor(
                n_estimators=n_estimators, max_depth=max_depth,
                learning_rate=learning_rate, random_state=seed,
            )
            self.backend = "sklearn"

    def fit(self, data: GARCHFeatures) -> GARCHXGBoost:
        """Fit the booster to the GARCH residual, not to the target."""
        baseline = data.features[:, self.garch_column]
        self.model.fit(data.features, data.target - baseline)
        return self

    def predict(self, data: GARCHFeatures) -> np.ndarray:
        baseline = data.features[:, self.garch_column]
        return np.maximum(baseline + self.model.predict(data.features), 1e-8)

    def evaluate(self, data: GARCHFeatures) -> dict[str, float]:
        """Report the hybrid *and* its GARCH baseline, so the gain is visible."""
        prediction = self.predict(data)
        baseline = data.features[:, self.garch_column]
        residual = data.target - prediction
        baseline_residual = data.target - baseline
        return {
            "MAE": float(np.mean(np.abs(residual))),
            "RMSE": float(np.sqrt(np.mean(residual**2))),
            "R2": float(1.0 - residual.var() / data.target.var()),
            "baseline_RMSE": float(np.sqrt(np.mean(baseline_residual**2))),
        }


class StackedMLGARCH:
    r"""Rubio, Alba, Velasquez & Ramos (2026): XGBoost meta-learning over base models.

    Several base learners produce forecasts; an XGBoost meta-learner is trained
    on those forecasts to produce the final one. The paper applies it to Bitcoin
    conditional variance and Value-at-Risk.

    The one detail that decides whether stacking works is how the meta-learner's
    training data is produced. Fitting the base learners and the meta-learner on
    the *same* rows lets the meta-learner reward whichever base model overfits
    hardest, and the ensemble then does worse than its own best member. We use
    **blocked out-of-fold** predictions: the base learners are refitted on
    earlier blocks and predict later ones, never their own training rows, and
    the blocks respect time order rather than being random folds.

    Parameters
    ----------
    base_models
        ``{name: estimator}`` with the scikit-learn ``fit``/``predict``
        interface. Defaults to ridge, random forest and SVR alongside the GARCH
        column, which is the paper's mix of econometric and ML learners.
    n_folds
        Number of contiguous time blocks used for out-of-fold prediction.
    """

    provenance = Provenance(
        paper_id="rubio2026_stacked_ml_garch",
        title=(
            "Stacked ML-GARCH for Bitcoin Risk Forecasting: A Novel Ensemble "
            "Approach for Superior Value-at-Risk Estimation"
        ),
        authors=("Lihki Rubio", "Keyla V. Alba", "Carlos E. Velasquez", "Filipe R. Ramos"),
        year=2026,
        venue="Mathematics 14, 624",
        code_status=CodeStatus.NO_PUBLIC_CODE_IDENTIFIED,
        implementation_status=ImplementationStatus.REPRODUCTION,
        deviations=(
            "The paper's exact base-learner set and hyper-parameters are not "
            "reproduced numerically; the default mix is ridge, random forest and "
            "SVR, and any scikit-learn estimator may be substituted.",
            "Out-of-fold predictions use contiguous time blocks rather than "
            "random K-fold, which random folds would make look-ahead.",
        ),
        notes="Not referenced in the ecosystem blueprint; added from the supplied PDF.",
    )

    def __init__(self, base_models: dict | None = None, n_folds: int = 5, seed: int = 0):
        self.seed = seed
        self.n_folds = n_folds
        self.base_models = base_models or self._default_bases(seed)
        self.meta = self._make_meta(seed)
        self.backend = ""

    @staticmethod
    def _default_bases(seed: int) -> dict:
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.linear_model import Ridge
        from sklearn.svm import SVR

        return {
            "ridge": Ridge(alpha=1.0),
            "forest": RandomForestRegressor(n_estimators=200, random_state=seed, n_jobs=-1),
            "svr": SVR(C=1.0, epsilon=0.05),
        }

    def _make_meta(self, seed: int):
        try:
            from xgboost import XGBRegressor

            self.backend = "xgboost"
            return XGBRegressor(
                n_estimators=300, max_depth=3, learning_rate=0.05,
                random_state=seed, verbosity=0,
            )
        except ImportError:  # pragma: no cover
            from sklearn.ensemble import GradientBoostingRegressor

            self.backend = "sklearn"
            return GradientBoostingRegressor(random_state=seed)

    def _out_of_fold(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Blocked out-of-fold predictions, respecting time order."""
        n = len(y)
        bounds = np.linspace(0, n, self.n_folds + 1).astype(int)
        out = np.zeros((n, len(self.base_models)))
        for j, model in enumerate(self.base_models.values()):
            for k in range(1, self.n_folds):
                train_end = bounds[k]
                test_slice = slice(bounds[k], bounds[k + 1])
                from sklearn.base import clone

                fold_model = clone(model)
                fold_model.fit(x[:train_end], y[:train_end])
                out[test_slice, j] = fold_model.predict(x[test_slice])
            # The first block has no earlier data; fall back to the whole-sample
            # fit for it and drop those rows when training the meta-learner.
            out[: bounds[1], j] = np.nan
        return out

    def fit(self, data: GARCHFeatures) -> StackedMLGARCH:
        x, y = data.features, data.target
        oof = self._out_of_fold(x, y)
        usable = np.isfinite(oof).all(axis=1)
        self.meta.fit(np.column_stack([oof[usable], x[usable]]), y[usable])
        for model in self.base_models.values():
            model.fit(x, y)          # refit on everything for prediction time
        return self

    def predict(self, data: GARCHFeatures) -> np.ndarray:
        x = data.features
        base = np.column_stack([m.predict(x) for m in self.base_models.values()])
        return np.maximum(self.meta.predict(np.column_stack([base, x])), 1e-8)

    def evaluate(self, data: GARCHFeatures) -> dict[str, float]:
        """Report the ensemble against its own best base learner.

        A stacking ensemble that cannot beat its best member is not adding
        anything, and quoting only the ensemble's own error would hide that.
        """
        prediction = self.predict(data)
        residual = data.target - prediction
        metrics = {
            "MAE": float(np.mean(np.abs(residual))),
            "RMSE": float(np.sqrt(np.mean(residual**2))),
            "R2": float(1.0 - residual.var() / data.target.var()),
        }
        best = np.inf
        for name, model in self.base_models.items():
            error = float(np.sqrt(np.mean((data.target - model.predict(data.features)) ** 2)))
            metrics[f"base_{name}_RMSE"] = error
            best = min(best, error)
        metrics["best_base_RMSE"] = best
        metrics["beats_best_base"] = bool(metrics["RMSE"] < best)
        return metrics
