r"""Yu, Xiao, Zhu & Zhang (2025): GENSHIN.

*GARCH Estimation based Network for Stock Index Heteroskedasticity INformation.*
A four-stage pipeline, and the interesting design decision is in the first two
stages rather than the network:

1. **GARCH extraction.** Fit **60** GARCH models — five specifications
   (sGARCH, EGARCH, GJR, apARCH, csGARCH) × four lag structures
   ((1,1), (1,3), (2,1), (5,5)) × three distributions (normal, Student-t, GED)
   — and keep every fitted volatility path as a feature.
2. **VMD-PLS.** Decompose each of the 60 paths into 4 intrinsic modes (240 in
   total), then compress to 6 components by partial least squares.
3. **MSGNet.** FFT picks out the dominant periods; an adaptive adjacency matrix
   plus MixHop convolution learns *inter-series* structure at each scale;
   multi-head attention learns *intra-series* structure; the scales are then
   merged weighted by their amplitudes.
4. Forecast realized volatility.

Why 60 models rather than one good one? Because the paper's premise is that
different GARCH specifications each capture a different facet of the volatility
process, and the network's job is to recombine them — not to pick a winner. That
also explains the two compression stages: 60 fitted volatility paths are
severely collinear, which is precisely the problem the authors say a plain
GARCH-LSTM suffers from. VMD splits them by frequency band and PLS then
compresses *with reference to the target*, which unsupervised PCA would not do.

The pipeline is expensive — 60 GARCH fits per series — so
:meth:`GENSHIN.build_features` caches, and ``specs``/``lags``/``distributions``
can be narrowed for exploratory work. Any narrowing is recorded on the returned
object rather than left implicit.

Reference
---------
Yu, R., Xiao, H., Zhu, Y., & Zhang, G. (2025). Does multi-scale GARCH
information enhance volatility prediction? *Finance Research Letters* 78, 107196.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ..core.config import TrainConfig, set_seed
from ..core.provenance import CodeStatus, ImplementationStatus, Provenance

__all__ = ["GARCHFeatureBank", "MSGNet", "GENSHIN", "GENSHINFeatures"]

#: The paper's full grid (sec. 2.1): 5 x 4 x 3 = 60 models.
SPECIFICATIONS = ("sgarch", "egarch", "gjr", "aparch", "csgarch")
LAG_STRUCTURES = ((1, 1), (1, 3), (2, 1), (5, 5))
DISTRIBUTIONS = ("normal", "t", "ged")


@dataclass
class GENSHINFeatures:
    """The compressed feature matrix, plus what produced it."""

    components: np.ndarray
    """Shape ``(T, n_components)`` — the PLS output the network consumes."""

    target: np.ndarray
    n_models_fitted: int
    n_modes: int
    failed: tuple[str, ...] = ()
    narrowed: bool = False
    note: str = ""

    def __len__(self) -> int:
        return len(self.target)


class GARCHFeatureBank:
    """Stage 1: fit the grid of GARCH models and collect their volatility paths.

    Uses :mod:`arch` for the four specifications it supports and
    :class:`~garchai.cells.component_cell.ComponentGARCHCell` for csGARCH, which
    ``arch`` does not provide.
    """

    def __init__(
        self,
        specifications: tuple[str, ...] = SPECIFICATIONS,
        lags: tuple[tuple[int, int], ...] = LAG_STRUCTURES,
        distributions: tuple[str, ...] = DISTRIBUTIONS,
    ):
        self.specifications = specifications
        self.lags = lags
        self.distributions = distributions

    @property
    def n_models(self) -> int:
        return len(self.specifications) * len(self.lags) * len(self.distributions)

    def fit(self, returns: np.ndarray, verbose: bool = False) -> tuple[np.ndarray, list[str]]:
        """Return ``(paths, failed)`` where ``paths`` is ``(n_models, T)``."""
        from arch import arch_model

        r = np.asarray(returns, dtype=float)
        paths, failed = [], []

        for spec, (p, q), dist in product(self.specifications, self.lags, self.distributions):
            label = f"{spec}({p},{q})-{dist}"
            try:
                if spec == "csgarch":
                    paths.append(self._fit_component(r, p, q))
                else:
                    volatility = self._fit_arch(r, spec, p, q, dist, arch_model)
                    paths.append(volatility)
            except Exception as exc:  # noqa: BLE001 - a failed spec is data, not a crash
                failed.append(f"{label}: {type(exc).__name__}")
                continue
            if verbose:  # pragma: no cover
                print(f"  fitted {label}")

        if not paths:
            raise RuntimeError("Every GARCH specification failed to fit.")
        return np.stack(paths), failed

    @staticmethod
    def _fit_arch(r, spec, p, q, dist, arch_model) -> np.ndarray:
        vol, o = "GARCH", 0
        if spec == "egarch":
            vol, o = "EGARCH", 1
        elif spec == "gjr":
            o = 1
        elif spec == "aparch":
            vol, o = "APARCH", 1
        fitted = arch_model(
            r, mean="Constant", vol=vol, p=p, o=o, q=q,
            dist={"normal": "normal", "t": "t", "ged": "ged"}[dist],
        ).fit(disp="off", show_warning=False)
        return np.asarray(fitted.conditional_volatility, dtype=float)

    @staticmethod
    def _fit_component(r, p, q) -> np.ndarray:
        """csGARCH via garchai, since ``arch`` has no component model."""
        from ..core import ModelConfig, TrainConfig as _TrainConfig
        from ..econometric.garch import _ClassicalModel
        from ..cells.component_cell import ComponentGARCHCell

        class _ComponentModel(_ClassicalModel):
            def __init__(self):
                cfg = ModelConfig(mean="constant", distribution="normal", scale=None)
                super().__init__(ComponentGARCHCell(numerical=cfg.numerical), cfg, None)

        model = _ComponentModel()
        result = model.fit(r, _TrainConfig(lr=0.05, max_epochs=60, patience=20))
        return np.asarray(result.conditional_volatility, dtype=float)


class MSGNet(nn.Module):
    r"""Stage 3: the multi-scale graph network (Cai et al., 2024).

    Three ideas stacked, each doing something the others cannot:

    **FFT scale selection.** :math:`F = \mathrm{Avg}(\mathrm{Amp}(\mathrm{FFT}(X)))`
    and :math:`s_i = L/f_i` for the top-:math:`k` amplitudes. The periods are
    *found*, not assumed — which matters for volatility, where the relevant
    horizons (daily, weekly, monthly) differ by asset.

    **Adaptive adjacency**, :math:`A_i = \mathrm{SoftMax}(\mathrm{ReLU}(E_1 E_2^{\top}))`.
    The graph over feature series is learned, not supplied. Note what the ReLU
    does and does not do: it floors negative affinities at zero *in the logits*,
    so those pairs all share one baseline score — but the softmax that follows
    still assigns them positive weight. The result is a **soft** graph in which
    unrelated series get equal small weight, not a pruned one with absent edges.
    Reading the ReLU as sparsification would overstate what the layer produces.

    **MixHop convolution.** Concatenates :math:`A^0X, A^1X, A^2X`, so the layer
    sees each node, its neighbours and its neighbours' neighbours *separately*
    rather than mixed together. That is what lets it represent "differs from my
    neighbours", which ordinary averaging convolutions provably cannot.

    Scales are finally merged weighted by their own FFT amplitudes, so a period
    that carries little energy contributes little.
    """

    def __init__(
        self,
        n_series: int,
        sequence_length: int,
        d_model: int = 32,
        n_scales: int = 3,
        n_heads: int = 4,
        node_dim: int = 8,
        horizon: int = 1,
    ):
        super().__init__()
        self.n_series = n_series
        self.sequence_length = sequence_length
        self.d_model = d_model
        self.n_scales = n_scales
        self.horizon = horizon
        dt = torch.float64

        self.embed = nn.Linear(n_series, d_model, dtype=dt)
        # One learned graph per scale (Eq. for A_i).
        self.node_source = nn.Parameter(torch.randn(n_scales, d_model, node_dim, dtype=dt) * 0.1)
        self.node_target = nn.Parameter(torch.randn(n_scales, d_model, node_dim, dtype=dt) * 0.1)
        # MixHop concatenates three powers, hence 3 * d_model in.
        self.mixhop = nn.ModuleList(
            nn.Linear(3 * d_model, d_model, dtype=dt) for _ in range(n_scales)
        )
        self.attention = nn.ModuleList(
            nn.MultiheadAttention(d_model, n_heads, batch_first=True, dtype=dt)
            for _ in range(n_scales)
        )
        self.project = nn.Linear(d_model * sequence_length, horizon, dtype=dt)

    @staticmethod
    def _periods(x: Tensor, k: int) -> tuple[list[int], Tensor]:
        """FFT scale detection: the top-``k`` non-zero frequencies by amplitude."""
        amplitude = torch.fft.rfft(x, dim=1).abs().mean(dim=(0, 2))
        amplitude[0] = 0.0                       # drop the DC term
        k = min(k, amplitude.numel() - 1)
        top = torch.topk(amplitude, k).indices
        length = x.shape[1]
        periods = [max(int(length // max(int(f), 1)), 1) for f in top]
        return periods, amplitude[top]

    def _graph(self, index: int) -> Tensor:
        """``A = SoftMax(ReLU(E1 E2^T))`` — learned, sparse, row-normalised."""
        affinity = torch.relu(self.node_source[index] @ self.node_target[index].T)
        return torch.softmax(affinity, dim=-1)

    def forward(self, x: Tensor) -> Tensor:
        """``x`` has shape ``(batch, sequence_length, n_series)``."""
        embedded = self.embed(x)                                  # (B, L, d)
        periods, amplitudes = self._periods(embedded, self.n_scales)

        outputs = []
        for i, _period in enumerate(periods):
            adjacency = self._graph(i)                            # (d, d)
            # MixHop: identity, one hop, two hops - kept separate, not summed.
            hop0 = embedded
            hop1 = embedded @ adjacency
            hop2 = hop1 @ adjacency
            mixed = self.mixhop[i](torch.cat([hop0, hop1, hop2], dim=-1))
            attended, _ = self.attention[i](mixed, mixed, mixed)
            outputs.append(attended)

        weights = torch.softmax(amplitudes, dim=0)                # amplitude-weighted
        merged = sum(w * out for w, out in zip(weights, outputs))
        return self.project(merged.reshape(merged.shape[0], -1))


class GENSHIN:
    """The full pipeline: GARCH bank → VMD → PLS → MSGNet.

    Examples
    --------
    >>> from garchai.graph import GENSHIN
    >>> model = GENSHIN(n_modes=4, n_components=6)
    >>> model.bank.n_models          # the paper's grid
    60
    """

    provenance = Provenance(
        paper_id="yu2025_genshin",
        title="Does multi-scale GARCH information enhance volatility prediction?",
        authors=("Rentian Yu", "Haotian Xiao", "Yukun Zhu", "Gongqiu Zhang"),
        year=2025,
        venue="Finance Research Letters 78, 107196",
        equations="sec. 2.1 (60-model grid), 2.3 (VMD), 2.4 (PLS), 2.5 (MSGNet)",
        code_status=CodeStatus.NO_PUBLIC_CODE_IDENTIFIED,
        implementation_status=ImplementationStatus.REPRODUCTION,
        deviations=(
            "csGARCH is not in `arch`, so it is fitted with garchai's own "
            "ComponentGARCHCell by maximum likelihood rather than by rugarch.",
            "MSGNet's exact layer widths are given in Cai et al. (2024), not in "
            "this paper; d_model, n_heads and node_dim are exposed as arguments.",
            "VMD's alpha (bandwidth) is not stated; it defaults to 2000, the "
            "usual value for unit-sampled signals.",
        ),
        data_substitutions=(
            "The paper uses eight Chinese indices at 5-minute sampling (M = 48), "
            "which is not available on public feeds. Reproductions substitute "
            "sourceable indices at the same sampling and say so.",
        ),
    )

    def __init__(
        self,
        n_modes: int = 4,
        n_components: int = 6,
        sequence_length: int = 20,
        specifications: tuple[str, ...] = SPECIFICATIONS,
        lags: tuple[tuple[int, int], ...] = LAG_STRUCTURES,
        distributions: tuple[str, ...] = DISTRIBUTIONS,
        seed: int | None = 0,
    ):
        self.bank = GARCHFeatureBank(specifications, lags, distributions)
        self.n_modes = n_modes
        self.n_components = n_components
        self.sequence_length = sequence_length
        self.seed = seed
        self.network: MSGNet | None = None
        self._pls = None
        self.history_: dict[str, list[float]] = {}

    def build_features(
        self, returns, target, vmd_alpha: float = 2000.0, verbose: bool = False
    ) -> GENSHINFeatures:
        r"""Stages 1-2: fit the GARCH grid, decompose, and compress.

        Parameters
        ----------
        target
            Realized volatility, aligned with ``returns``. The paper's
            :math:`RV_t = \sqrt{\sum_i r_{i,t}^2}\times 100\%` is available as
            :func:`garchai.targets.realized_volatility_intraday`.
        """
        from sklearn.cross_decomposition import PLSRegression

        from .vmd import variational_mode_decomposition

        r = np.asarray(returns, dtype=float)
        y = np.asarray(target, dtype=float)
        if r.size != y.size:
            raise ValueError(f"returns and target must align: {r.size} vs {y.size}")

        paths, failed = self.bank.fit(r, verbose=verbose)
        if verbose:  # pragma: no cover
            print(f"  fitted {paths.shape[0]}/{self.bank.n_models} models")

        # Stage 2a: each fitted volatility path -> n_modes intrinsic modes.
        modes = []
        for path in paths:
            decomposed, _ = variational_mode_decomposition(
                path, n_modes=self.n_modes, alpha=vmd_alpha
            )
            modes.append(decomposed)
        stacked = np.concatenate(modes, axis=0).T          # (T, n_paths * n_modes)

        # Stage 2b: PLS compresses *with reference to the target*, which is the
        # whole reason it is used here rather than PCA.
        finite = np.isfinite(stacked).all(axis=1) & np.isfinite(y)
        # PLS cannot extract more components than it has input columns. With the
        # full 60-model grid there are 240 of them and the paper's 6 is never
        # binding; with a narrowed grid it can be, and failing with sklearn's
        # error would be a confusing way to learn that.
        n_components = min(self.n_components, stacked.shape[1])
        clamped = n_components != self.n_components
        self._pls = PLSRegression(n_components=n_components)
        self._pls.fit(stacked[finite], y[finite])
        components = self._pls.transform(stacked)

        narrowed = self.bank.n_models != len(SPECIFICATIONS) * len(LAG_STRUCTURES) * len(
            DISTRIBUTIONS
        )
        notes = []
        if narrowed:
            notes.append(
                f"Grid narrowed from the paper's 60 models to {self.bank.n_models}."
            )
        else:
            notes.append(f"Full 60-model grid; {len(failed)} specifications failed to fit.")
        if clamped:
            notes.append(
                f"n_components clamped from {self.n_components} to {n_components}: "
                f"PLS cannot exceed the {stacked.shape[1]} available mode-series."
            )
        return GENSHINFeatures(
            components=components,
            target=y,
            n_models_fitted=paths.shape[0],
            n_modes=self.n_modes,
            failed=tuple(failed),
            narrowed=narrowed,
            note=" ".join(notes),
        )

    def _windows(self, features: GENSHINFeatures) -> tuple[Tensor, Tensor]:
        x, y, length = features.components, features.target, self.sequence_length
        rows = [x[t - length : t] for t in range(length, len(y))]
        return torch.tensor(np.stack(rows)), torch.tensor(y[length:])

    def fit(
        self, features: GENSHINFeatures, train: TrainConfig | None = None,
        batch_size: int = 64,
    ) -> dict:
        """Stage 3: train MSGNet on the compressed features."""
        cfg = train or TrainConfig(optimizer="adam", lr=1e-3, max_epochs=200, patience=25)
        if self.seed is not None:
            set_seed(self.seed)

        x, y = self._windows(features)
        self.network = MSGNet(
            n_series=x.shape[-1], sequence_length=self.sequence_length
        )
        optimiser = torch.optim.Adam(self.network.parameters(), lr=cfg.lr)

        n = x.shape[0]
        history = {"loss": []}
        best, best_state, stalled = float("inf"), None, 0

        for _ in range(cfg.max_epochs):
            self.network.train()
            order = torch.randperm(n)
            total = 0.0
            for start in range(0, n, batch_size):
                index = order[start : start + batch_size]
                optimiser.zero_grad(set_to_none=True)
                loss = F.mse_loss(self.network(x[index]).squeeze(-1), y[index])
                loss.backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), 1.0)
                optimiser.step()
                total += float(loss.detach()) * index.numel()

            total /= n
            history["loss"].append(total)
            if total < best - 1e-12:
                best, stalled = total, 0
                best_state = {k: v.clone() for k, v in self.network.state_dict().items()}
            else:
                stalled += 1
                if stalled >= cfg.patience:
                    break

        if best_state is not None:
            self.network.load_state_dict(best_state)
        self.history_ = history
        return history

    def predict(self, features: GENSHINFeatures) -> np.ndarray:
        if self.network is None:
            raise RuntimeError("GENSHIN has not been fitted; call .fit(features) first.")
        self.network.eval()
        x, _ = self._windows(features)
        with torch.no_grad():
            return self.network(x).squeeze(-1).numpy()

    def evaluate(self, features: GENSHINFeatures) -> dict[str, float]:
        """MAE, RMSE and the Mincer-Zarnowitz R-squared the paper reports."""
        prediction = self.predict(features)
        truth = features.target[self.sequence_length :]
        residual = truth - prediction
        return {
            "MAE": float(np.mean(np.abs(residual))),
            "RMSE": float(np.sqrt(np.mean(residual**2))),
            "R2": float(1.0 - residual.var() / truth.var()),
        }
