# Changelog

All notable changes to `garchai`. Follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and
[semantic versioning](https://semver.org/).

## [0.1.1] — 2026-10-29

### Fixed

- **`forecast()` failed for every neural and structural model.**
  `VolatilityModel.forecast` stepped `self.cell` through the `VarianceCell`
  interface (`initial_state`, `step`), which only the classical models
  (GARCH, GJR-GARCH, EGARCH) have. GARCHNN and StructuralGARCHLSTM expose a
  kernel there, and the other neural families a network, so `forecast()`
  raised `AttributeError` or `TypeError` for all of them.
  - Models without a `VarianceCell` now go through
    `_forecast_by_recursion`. The one-step forecast is exact for every model:
    it is read off the model's own `filter`, since `h_{T+1}` depends only on
    information up to `T`.
  - GARCHNN and StructuralGARCHLSTM also forecast multiple steps, by Monte
    Carlo through the same recursion, vectorised across paths. On a GARCH(1,1)
    kernel the simulated mean reproduces the closed-form decay.
  - The other neural families raise a clear `NotImplementedError` for
    `horizon > 1` instead of crashing; `horizon=1` works.
- The Zhao models' `filter` is now built on a shared `_step`, used by both
  filtering and forecasting. Variance paths are bit-identical to 0.1.0.
- `to_tensor` copies its input. pandas >= 3 hands out read-only views, which
  `torch.as_tensor` would otherwise alias, with a warning.

## [0.1.0] — 2026-07

First release.
