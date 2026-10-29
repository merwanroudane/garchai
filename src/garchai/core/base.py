r"""The base volatility model: mean + variance cell + distribution, fitted jointly.

This class owns the parts a :class:`~garchai.cells.base.VarianceCell`
deliberately does not: the data, the scaling, the mean model, the likelihood and
the optimiser. Keeping them separate is what lets the same cell be estimated by
maximum likelihood here and dropped inside a neural architecture in
:mod:`garchai.structural` without change.

Estimation is by maximum likelihood over the *joint* parameter vector — mean,
variance and distribution shape together. Two-stage estimation (fit the mean,
then fit the variance to its residuals) is consistent but inefficient, and it is
not what the GARCH literature reports.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

from .config import ModelConfig, TrainConfig, resolve_scale, set_seed

if TYPE_CHECKING:  # avoids a cycle: cells.base -> core.config -> core.base
    from ..cells.base import VarianceCell
    from ..distributions.base import Distribution
    from ..mean.models import MeanModel
from .provenance import Provenance
from .results import FitResult, ForecastResult, to_numpy, to_tensor


class VolatilityModel(nn.Module, ABC):
    """Base class for a fitted-by-likelihood conditional volatility model.

    Parameters
    ----------
    config
        Mean specification, distribution, dtype, scaling and numerical guards.

    Attributes set by :meth:`fit`
    -----------------------------
    ``conditional_variance_``, ``conditional_volatility_``,
    ``standardized_residuals_``, ``loglikelihood_``, ``result_``,
    ``data_scale_``.
    """

    provenance: Provenance | None = None

    def __init__(self, config: ModelConfig | None = None):
        super().__init__()
        self.config = config or ModelConfig()
        if self.config.seed is not None:
            set_seed(self.config.seed)
        self.data_scale_: float = 1.0
        self.result_: FitResult | None = None
        self._fitted_returns: Tensor | None = None

    # -- to be provided by subclasses --------------------------------------- #
    @property
    @abstractmethod
    def cell(self) -> VarianceCell:
        """The variance recursion."""

    @property
    @abstractmethod
    def mean_model(self) -> MeanModel:
        """The conditional-mean specification."""

    @abstractmethod
    def distribution(self) -> Distribution:
        """The innovation distribution, built from current shape parameters."""

    # -- likelihood --------------------------------------------------------- #
    def filter(self, r: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Return ``(h, eps, mu)`` for a return series on the internal scale."""
        eps, mu = self.mean_model.residuals(r)
        h, _ = self.cell.filter(eps)
        return h, eps, mu

    def negative_loglikelihood(self, r: Tensor) -> Tensor:
        """Joint negative log-likelihood, excluding mean-model pre-sample rows."""
        h, eps, _ = self.filter(r)
        start = self.mean_model.n_presample
        return self.distribution().nll(eps[start:], h[start:], reduction="sum")

    # -- estimation --------------------------------------------------------- #
    def fit(
        self,
        returns,
        train: TrainConfig | None = None,
        callback=None,
    ) -> FitResult:
        r"""Estimate the model by maximum likelihood.

        Parameters
        ----------
        returns
            The return series :math:`r_t`. Not the innovations — the mean model
            is what turns one into the other.
        train
            Optimiser settings. Defaults follow the protocol shared by Zhao et
            al. (2024) and Wei et al. (2025): Adam, learning-rate halving on
            plateau, early stopping with patience 20.
        callback
            Optional ``f(epoch, loss)`` called each epoch.

        Notes
        -----
        Optimisation runs on a rescaled series when ``config.scale`` asks for
        it, because daily returns of order 1e-2 drive :math:`\omega` to 1e-6 and
        make the likelihood surface badly conditioned. Everything in the
        returned :class:`~garchai.core.results.FitResult` is transformed back to
        the original scale; ``result.scale`` records the factor.
        """
        cfg = train or TrainConfig()
        r_raw = to_tensor(returns, dtype=self.config.torch_dtype)
        self.data_scale_ = resolve_scale(r_raw, self.config.scale)
        r = r_raw * self.data_scale_
        self._fitted_returns = r

        self.to(self.config.torch_dtype)
        optimiser = self._build_optimiser(cfg)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimiser, factor=cfg.lr_factor, patience=cfg.lr_patience, min_lr=cfg.min_lr
        )

        n_val = int(round(cfg.validation_split * r.numel()))
        r_train = r[: r.numel() - n_val] if n_val else r
        r_val = r[r.numel() - n_val :] if n_val else None

        history: dict[str, list[float]] = {"loss": [], "val_loss": [], "lr": []}
        best_loss, best_state, since_improvement = float("inf"), None, 0
        message = "maximum epochs reached"

        for epoch in range(cfg.max_epochs):
            optimiser.zero_grad(set_to_none=True)
            loss = self.negative_loglikelihood(r_train)
            if not torch.isfinite(loss):
                message = f"non-finite loss at epoch {epoch}; stopped"
                break
            loss.backward()
            if self.config.numerical.grad_clip is not None:
                nn.utils.clip_grad_norm_(self.parameters(), self.config.numerical.grad_clip)
            optimiser.step()

            with torch.no_grad():
                monitored = (
                    float(self.negative_loglikelihood(r_val)) if r_val is not None
                    else float(loss)
                )
            history["loss"].append(float(loss.detach()))
            history["val_loss"].append(monitored if r_val is not None else float("nan"))
            history["lr"].append(optimiser.param_groups[0]["lr"])
            scheduler.step(monitored)
            if callback is not None:
                callback(epoch, monitored)

            if monitored < best_loss - 1e-10:
                best_loss, since_improvement = monitored, 0
                best_state = {k: v.detach().clone() for k, v in self.state_dict().items()}
            else:
                since_improvement += 1
                if since_improvement >= cfg.patience:
                    message = f"early stopping at epoch {epoch} (patience {cfg.patience})"
                    break

        if best_state is not None:
            self.load_state_dict(best_state)
        if cfg.refine_with_lbfgs:
            message += "; refined with LBFGS"
            self._refine_lbfgs(r_train, cfg)

        self.result_ = self._build_result(r, history, message)
        return self.result_

    def _build_optimiser(self, cfg: TrainConfig) -> torch.optim.Optimizer:
        params = [p for p in self.parameters() if p.requires_grad]
        if not params:
            raise ValueError("Model has no trainable parameters.")
        if cfg.optimizer == "adam":
            return torch.optim.Adam(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
        if cfg.optimizer == "adamw":
            return torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
        if cfg.optimizer == "lbfgs":
            return torch.optim.LBFGS(params, lr=cfg.lr, max_iter=cfg.lbfgs_max_iter)
        raise ValueError(f"Unknown optimizer {cfg.optimizer!r}.")

    def _refine_lbfgs(self, r: Tensor, cfg: TrainConfig) -> None:
        """Polish an Adam solution with LBFGS.

        Adam gets close quickly but stalls near the optimum; LBFGS uses
        curvature and tightens the last few digits of the likelihood, which
        matters when comparing models by AIC.
        """
        params = [p for p in self.parameters() if p.requires_grad]
        optimiser = torch.optim.LBFGS(params, lr=1.0, max_iter=cfg.lbfgs_max_iter,
                                      line_search_fn="strong_wolfe")

        def closure() -> Tensor:
            optimiser.zero_grad(set_to_none=True)
            loss = self.negative_loglikelihood(r)
            if torch.isfinite(loss):
                loss.backward()
            return loss

        try:
            optimiser.step(closure)
        except RuntimeError:  # pragma: no cover - LBFGS can fail on flat regions
            pass

    # -- results ------------------------------------------------------------ #
    def _build_result(self, r: Tensor, history: dict, message: str) -> FitResult:
        scale = self.data_scale_
        with torch.no_grad():
            h, eps, mu = self.filter(r)
            start = self.mean_model.n_presample
            loglik = -float(self.distribution().nll(eps[start:], h[start:], reduction="sum"))
            z = eps * torch.rsqrt(h)

            params, persistence, long_run = self._variance_diagnostics()

        # Undo the scaling: variance carries scale^2, mean and residuals scale^1,
        # and standardized residuals are scale-free by construction.
        return FitResult(
            conditional_variance=to_numpy(h) / scale**2,
            conditional_volatility=to_numpy(torch.sqrt(h)) / scale,
            residuals=to_numpy(eps) / scale,
            standardized_residuals=to_numpy(z),
            conditional_mean=to_numpy(mu) / scale,
            loglikelihood=loglik,
            n_obs=int(r.numel()) - start,
            n_params=sum(p.numel() for p in self.parameters() if p.requires_grad),
            n_presample=start,
            parameters=params,
            mean_parameters={k: float(v) / scale for k, v in
                             self.mean_model.parameter_dict().items() if v.numel() == 1},
            distribution_parameters=self._distribution_parameters(),
            persistence=persistence,
            long_run_variance=long_run / scale**2 if np.isfinite(long_run) else long_run,
            scale=scale,
            history=history,
            converged="non-finite" not in message,
            message=message,
        )

    def _distribution_parameters(self) -> dict[str, float]:
        return {}

    def _variance_diagnostics(self) -> tuple[dict[str, float], float, float]:
        """Econometric parameters, persistence and long-run variance.

        Reads them off ``self.cell`` by default. Structural models whose "cell"
        is a recurrent network rather than a GARCH recursion override this to
        report the econometric sub-component instead, which is the part that
        stays interpretable after joint training.
        """
        cell = self.cell
        params = {
            k: float(v.detach())
            for k, v in cell.transformed_parameters().items()
            if v.numel() == 1
        }
        persistence = float(cell.persistence().detach())
        long_run = float(cell.unconditional_variance().detach())
        return params, persistence, long_run

    # -- forecasting -------------------------------------------------------- #
    def supports_analytic_forecast(self) -> bool:
        """Whether a closed-form multi-step variance forecast exists.

        Overridden to ``False`` by families where it does not — EGARCH being the
        obvious case, since forecasting :math:`h_{T+k}` requires
        :math:`E[\\exp(\\cdot)]` rather than :math:`\\exp(E[\\cdot])`.
        """
        return True

    def analytic_forecast_variance(self, h_next: Tensor, horizon: int) -> Tensor:
        r"""Closed-form multi-step variance forecast.

        For a GARCH-family recursion with persistence :math:`P` and long-run
        variance :math:`\bar\sigma^2 = \omega/(1-P)`, iterating the recursion
        under :math:`E[\epsilon_{T+j}^2 \mid \mathcal{F}_T] = h_{T+j}` gives

        .. math::
            E[h_{T+k} \mid \mathcal{F}_T]
              = \bar\sigma^2 + P^{\,k-1}\big(h_{T+1} - \bar\sigma^2\big).

        The forecast decays geometrically from the current variance toward the
        long-run level at rate ``P``, which is exactly why persistence is the
        number to look at when a model's forecasts look implausibly flat or
        implausibly persistent.
        """
        p = self.cell.persistence()
        long_run = self.cell.unconditional_variance()
        k = torch.arange(horizon, dtype=h_next.dtype, device=h_next.device)
        if not torch.isfinite(long_run) or float(p) >= 1.0:
            # Non-stationary: the variance does not mean-revert, so the best
            # forecast is a random walk in h.
            return h_next.expand(horizon).clone()
        return long_run + (p**k) * (h_next - long_run)

    def forecast(
        self,
        horizon: int = 1,
        method: str = "auto",
        n_simulations: int = 10_000,
        seed: int | None = None,
    ) -> ForecastResult:
        r"""Forecast the conditional variance ``horizon`` steps beyond the sample.

        Parameters
        ----------
        horizon
            Number of steps ahead.
        method
            ``"analytic"``, ``"simulation"``, or ``"auto"`` (analytic where the
            family supports it, simulation otherwise).

            Blueprint sec. 27 is explicit that a simulation forecast must never
            be silently substituted for an analytic one, so asking for
            ``"analytic"`` from a model that has no closed form raises rather
            than quietly switching. ``"auto"`` is the option that chooses for
            you, and the choice is recorded in ``result.method``.
        n_simulations
            Number of Monte Carlo paths when simulating.
        seed
            Seed for the simulation, for reproducibility.

        Returns
        -------
        ForecastResult
            Forecasts on the original data scale.
        """
        if horizon < 1:
            raise ValueError("horizon must be at least 1.")
        self._require_fit()
        r = self._fitted_returns
        assert r is not None

        if method == "auto":
            method = "analytic" if self.supports_analytic_forecast() else "simulation"
        elif method == "analytic" and not self.supports_analytic_forecast():
            raise ValueError(
                f"{type(self).__name__} has no closed-form multi-step variance "
                "forecast. Use method='simulation', or method='auto' to let the "
                "model choose. garchai will not silently substitute simulation "
                "for an analytic forecast."
            )

        from ..cells.base import VarianceCell

        with torch.no_grad():
            if not isinstance(self.cell, VarianceCell):
                # Neural and structural families carry their own recursion
                # (kernels, LSTMs, coefficient networks), not a VarianceCell.
                variance, simulations, method = self._forecast_by_recursion(
                    r, horizon, method, n_simulations, seed
                )
                return self._package_forecast(variance, simulations, horizon, method)

            h, eps, _ = self.filter(r)
            state = self.cell.initial_state(eps)
            for t in range(1, eps.numel()):
                _, state = self.cell.step(eps[t - 1], state)
            h_next, state_next = self.cell.step(eps[-1], state)

            if method == "analytic":
                variance = self.analytic_forecast_variance(h_next, horizon)
                simulations = None
            elif method == "simulation":
                variance, simulations = self._simulate_forecast(
                    h_next, state_next, horizon, n_simulations, seed
                )
            else:
                raise ValueError(f"Unknown forecast method {method!r}.")

        return self._package_forecast(variance, simulations, horizon, method)

    def _package_forecast(
        self, variance: Tensor, simulations: Tensor | None, horizon: int, method: str
    ) -> ForecastResult:
        """Undo the data scaling and wrap a variance forecast in a ForecastResult."""
        scale = self.data_scale_
        mu = self._forecast_mean(horizon) / scale
        try:
            persistence = float(self.cell.persistence())
        except (AttributeError, TypeError, NotImplementedError):
            persistence = float("nan")
        return ForecastResult(
            variance=to_numpy(variance) / scale**2,
            volatility=to_numpy(torch.sqrt(variance)) / scale,
            mean=mu,
            horizon=horizon,
            method=method,
            scale=scale,
            simulations=None if simulations is None else to_numpy(simulations) / scale**2,
            metadata={"persistence": persistence},
        )

    def _one_step_variance(self, r: Tensor) -> Tensor:
        r"""Exact :math:`h_{T+1}` for any model, read off its own ``filter``.

        The conditional variance is predictable — :math:`h_{T+1}` depends on
        information up to :math:`T` only — so filtering the sample extended by
        one placeholder return yields it exactly, whatever the placeholder is.
        This needs nothing from the model beyond ``filter``.
        """
        h, _, _ = self.filter(torch.cat([r, r.new_zeros(1)]))
        return h[-1]

    def _forecast_by_recursion(
        self,
        r: Tensor,
        horizon: int,
        method: str,
        n_simulations: int,
        seed: int | None,
    ) -> tuple[Tensor, Tensor | None, str]:
        """Forecast for models without a VarianceCell.

        The one-step forecast is exact for every model. Multi-step forecasts
        need a model-specific simulation of the recursion; families that have
        one override this method.
        """
        if horizon > 1:
            raise NotImplementedError(
                f"{type(self).__name__} has no multi-step forecast yet; use "
                "horizon=1, which is exact, or roll one-step forecasts forward."
            )
        return self._one_step_variance(r).reshape(1), None, method

    def _simulate_forecast(
        self,
        h_next: Tensor,
        state,
        horizon: int,
        n_simulations: int,
        seed: int | None,
    ) -> tuple[Tensor, Tensor]:
        """Monte Carlo the recursion forward, sampling innovations each step."""
        from ..cells.base import CellState

        generator = torch.Generator(device=h_next.device)
        generator.manual_seed(seed if seed is not None else torch.seed() % (2**31))

        dist = self.distribution()
        paths = torch.empty(n_simulations, horizon, dtype=h_next.dtype)
        states = [
            CellState(
                h=state.h.clone(),
                log_h=None if state.log_h is None else state.log_h.clone(),
            )
            for _ in range(n_simulations)
        ]
        current = h_next.expand(n_simulations).clone()

        uniform = torch.rand(n_simulations, horizon, generator=generator, dtype=h_next.dtype)
        for k in range(horizon):
            paths[:, k] = current
            if k == horizon - 1:
                break
            z = torch.tensor(
                [float(dist.ppf(float(u))) for u in uniform[:, k].clamp(1e-9, 1 - 1e-9)],
                dtype=h_next.dtype,
            )
            eps = z * torch.sqrt(current)
            nxt = []
            for i in range(n_simulations):
                h_i, states[i] = self.cell.step(eps[i], states[i])
                nxt.append(h_i)
            current = torch.stack(nxt)
        return paths.mean(dim=0), paths

    def _forecast_mean(self, horizon: int) -> np.ndarray:
        """Forecast of the conditional mean. Constant for the Phase-1 models."""
        params = self.mean_model.parameter_dict()
        level = float(params["mu"]) if "mu" in params else 0.0
        return np.full(horizon, level)

    # -- convenience -------------------------------------------------------- #
    @property
    def conditional_variance_(self) -> np.ndarray:
        return self._require_fit().conditional_variance

    @property
    def conditional_volatility_(self) -> np.ndarray:
        return self._require_fit().conditional_volatility

    @property
    def standardized_residuals_(self) -> np.ndarray:
        return self._require_fit().standardized_residuals

    @property
    def loglikelihood_(self) -> float:
        return self._require_fit().loglikelihood

    def _require_fit(self) -> FitResult:
        if self.result_ is None:
            raise RuntimeError("Model has not been fitted yet; call .fit(returns) first.")
        return self.result_

    def var(self, alpha: float = 0.05) -> np.ndarray:
        """In-sample Value-at-Risk path, as a return level."""
        res = self._require_fit()
        q = float(self.distribution().ppf(alpha))
        return res.conditional_mean + res.conditional_volatility * q

    def es(self, alpha: float = 0.05) -> np.ndarray:
        """In-sample expected-shortfall path, as a return level."""
        res = self._require_fit()
        e = float(self.distribution().expected_shortfall(alpha))
        return res.conditional_mean + res.conditional_volatility * e

    def summary(self) -> str:
        """A readable estimation summary."""
        res = self._require_fit()
        lines = [
            f"{type(self).__name__} estimation results",
            "=" * 58,
            f"  Mean model      : {type(self.mean_model).__name__}",
            f"  Distribution    : {self.distribution().name}",
            f"  Observations    : {res.n_obs}"
            + (f" (+{res.n_presample} pre-sample)" if res.n_presample else ""),
            f"  Log-likelihood  : {res.loglikelihood:.4f}",
            f"  AIC / BIC       : {res.aic:.4f} / {res.bic:.4f}",
            f"  Data scale      : {res.scale:g}",
            "",
            "  Variance parameters",
            "  " + "-" * 30,
        ]
        lines += [f"    {k:<12s} {v: .6f}" for k, v in res.parameters.items()]
        if res.mean_parameters:
            lines += ["", "  Mean parameters", "  " + "-" * 30]
            lines += [f"    {k:<12s} {v: .6f}" for k, v in res.mean_parameters.items()]
        if res.distribution_parameters:
            lines += ["", "  Distribution parameters", "  " + "-" * 30]
            lines += [f"    {k:<12s} {v: .6f}" for k, v in res.distribution_parameters.items()]
        lines += [
            "",
            f"  Persistence      : {res.persistence:.6f}",
            f"  Long-run variance: {res.long_run_variance:.6g}",
            f"  Status           : {res.message}",
        ]
        if self.provenance is not None:
            lines += ["", self.provenance.summary()]
        return "\n".join(lines)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        state = "fitted" if self.result_ is not None else "unfitted"
        return f"{type(self).__name__}(distribution={self.distribution().name}, {state})"
