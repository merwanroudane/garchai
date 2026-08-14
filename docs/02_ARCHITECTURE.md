# `garchai` — proposed package architecture

Phase 0 deliverable. This is the structure I propose to build against; it follows
master blueprint §15 with four deliberate departures, flagged and justified at
the end.

---

## 1. Distribution identity

- **PyPI name:** `garchai`
- **Import name:** `garchai`
- **Repository:** https://github.com/merwanroudane/garchai
- **Backend:** PyTorch (float64 by default for econometric likelihoods), NumPy /
  SciPy for classical estimation, `arch` as an optional adapter.

One-line identity, from the blueprint:

> A research-grade ecosystem for econometrically constrained, differentiable,
> AI-enhanced conditional heteroskedasticity models.

## 1b. Scope boundary against `hybridecon`

`hybridecon` (same author, MIT, installed locally at
`xtpmg/hybridaieco/hybridecon`) already owns a large part of the hybrid space.
`garchai` does **not** rebuild it. The division:

| Family | Owner | garchai's route |
|---|---|---|
| Feature hybrids — GARCH σ/z → LSTM, GRU, BiLSTM, CNN-BiLSTM, attention-LSTM | **hybridecon** | `adapters/hybridecon_adapter.py` |
| HAR-LSTM-GARCH, ARIMA/ARFIMA-LSTM, MIDAS-LSTM, KAN-GARCH-MIDAS, GAS-ATT-LSTM | **hybridecon** / `harlstmgarch` | adapter + benchmark |
| DeepBEKK, DeepVAR, DeepTVAR, DeepQuantileVaR, RegimeAwareLSTMGARCH | **hybridecon** | adapter + benchmark |
| Classical GARCH estimation | **`arch`**, cross-checked against **`tsgarch`** | `adapters/arch_adapter.py` |
| **Structural** (GARCH-NN, embedded cells, σ-cells) | **garchai** | built here |
| **Time-varying-parameter / Bayesian** (NeuralGARCH, RECH, RealRECH, GARCHNet) | **garchai** | built here |
| **Informed losses** (GINN) | **garchai** | built here |
| **Constrained neural covariance** (NeuralBEKK, NeuralDCC) | **garchai** | built here |
| **Graph multi-scale** (GENSHIN) | **garchai** | built here |
| **Amortized parameter inference** | **garchai** | built here |

Because reuse rights are unrestricted (author's own MIT code), `hybridecon`'s
`evaluate.model_confidence_set`, VaR backtest table and `viz` table helpers are
**adapted** into `garchai.evaluation` / `garchai.reporting` rather than rewritten.
The econometric estimators *are* re-implemented, because garchai needs autograd
and float64 end to end, which the NumPy originals cannot provide.

**Hard naming constraint.** `hybridecon.hybrids.GARCHLSTM` is feature-based.
`garchai` therefore never exports a bare `GARCHLSTM`. It exports
`StructuralGARCHLSTM` (Zhao), `EmbeddedGARCHLSTM` (Wei), `GARCHInformedLSTM`
(Xu), and `FeatureGARCHLSTM` (Kim & Won, adapter-backed). Importing both
packages in one session must never be ambiguous.

## 2. Repository layout

```
garchai/
├── pyproject.toml
├── README.md
├── LICENSE                       (MIT — see open question 3)
├── CITATION.cff
├── docs/
│   ├── researcher_guide/         ← the "start from zero" guide
│   ├── models/                   ← one page per model, 19-point template (§60)
│   └── reproductions/
├── examples/                     ← runnable, real-data, numbered 01..N
├── src/garchai/
│   ├── core/          base.py config.py state.py results.py registry.py
│   ├── constraints/   positivity.py stationarity.py simplex.py covariance.py
│   ├── distributions/ normal.py student_t.py skew_t.py ged.py base.py
│   ├── losses/        nll.py qlike.py informed.py pinball.py
│   ├── cells/         base.py garch_cell.py gjr_cell.py egarch_cell.py
│   │                  figarch_conv.py embedded_garch_lstm.py
│   │                  embedded_garch_gru.py sigma_lstm.py sigma_cell.py
│   ├── econometric/   garch.py gjr.py egarch.py aparch.py figarch.py
│   │                  realized_garch.py garch_midas.py bekk.py dcc.py
│   ├── structural/    garch_nn.py neural_garch.py rech.py realrech.py
│   │                  garchnet.py parameter_networks.py
│   ├── informed/      ginn.py garch_informed_lstm.py
│   ├── feature_hybrids/  garch_lstm.py garch_gru.py garch_tft.py
│   ├── residual_hybrids/ garch_xgboost.py har_lstm_garch.py
│   ├── ensembles/     stacking.py weighted.py multigarch_transformer.py
│   ├── multivariate/  neural_bekk.py neural_dcc.py dcc_mann.py covariance.py
│   ├── graph/         genshin.py graph_builder.py multiscale.py
│   ├── fuzzy/         garch_fis.py
│   ├── rl/            environment.py garch_ddqn_var.py rewards.py
│   ├── regimes/       changepoint.py structural_breaks.py regime_neural_garch.py
│   ├── panel/         hierarchical_garch.py panel_neural_garch.py
│   ├── bayes/         priors.py variational.py smc.py posterior.py
│   ├── mean/          zero.py constant.py ar.py arma.py neural.py
│   ├── targets/       realized.py proxies.py        ← NEW, see §5
│   ├── forecast/      engine.py analytic.py recursive.py simulation.py
│   ├── risk/          var.py es.py kupiec.py christoffersen.py dq.py
│   ├── evaluation/    metrics.py diebold_mariano.py giacomini_white.py
│   │                  mcs.py rolling.py experiment.py
│   ├── diagnostics/   residuals.py arch_lm.py calibration.py normality.py
│   ├── adapters/      arch_adapter.py mvgarch_adapter.py sklearn_adapter.py
│   ├── datasets/      loaders.py registry.py  + data/*.csv.gz   ← NEW, see §5
│   ├── reporting/     tables.py figures.py style.py             ← NEW, see §5
│   ├── papers/        registry.yaml reproduce.py configs/
│   └── resources/     repos.yaml papers.yaml licenses.yaml
└── tests/
    ├── unit/ gradient/ constraints/ numerical/
    ├── paper_reproduction/
    └── benchmarks/
```

## 3. The two primitives everything rests on

### 3.1 `VarianceCell` — the differentiable recurrence

```python
class VarianceCell(nn.Module):
    def transformed_parameters(self) -> dict[str, Tensor]: ...
    def initial_state(self, y: Tensor) -> CellState: ...
    def step(self, eps_prev, h_prev, state=None) -> tuple[Tensor, CellState]: ...
    def persistence(self) -> Tensor: ...
```

Non-negotiable rules, from blueprint §16 and §132:

- `h_prev` and `eps_prev` are **never** detached unless an algorithm explicitly
  requires truncated BPTT, and then only via `TruncationConfig`.
- Raw and transformed parameters are both exposed.
- The cell owns no data, no scaling and no mean model.

This single abstraction covers `GARCHCell`, `GJRCell`, `EGARCHCell`, the
FI-GARCH convolutional kernel of Zhao et al., the Wei et al. embedded cells, the
σ-cells, and the time-varying-parameter cells of Yin & Barucca.

### 3.2 `ConstraintTransform` — the econometric guardrail

```python
GARCHConstraint()     # ω = softplus(raw)+ε ;  (α,β) via (1−δ)·softmax slack
GJRConstraint()       # + α + γ/2 + β < 1
EGARCHConstraint()    # |β| < 1, log-variance clipped to max_log_variance
BEKKConstraint()      # Cholesky: diag = softplus(raw), H = L Lᵀ
DCCConstraint()       # a,b ≥ 0, a+b < 1; R_t symmetric, unit diagonal
```

Applied **at every time step** for time-varying parameters (blueprint §17.3).
Each family gets its own transform; GARCH(1,1) constraints are never forced onto
another family.

## 4. Model API (blueprint §38, §50)

```python
model = EmbeddedGARCHGRU(
    hidden_size=32,
    distribution="student-t",
    mean="constant",
    enforce_stationarity=True,
    dtype="float64",
    scale=100,
    seed=42,
)
res = model.fit(returns, validation_split=0.2)
fc  = model.forecast(horizon=10, method="recursive")
var = model.var(alpha=0.01)
print(model.summary())
```

Fitted attributes: `conditional_variance_`, `conditional_volatility_`,
`standardized_residuals_`, `loglikelihood_`, `econometric_parameters_`,
`neural_parameters_`, `parameter_paths_` (time-varying models only), `history_`,
`config_`, `data_scale_`.

Every model class carries provenance metadata (blueprint §43):

```python
EmbeddedGARCHGRU.provenance
# Provenance(paper_id='wei2025_garch_rnn',
#            paper_url='https://arxiv.org/abs/2504.09380',
#            equations='Eqs. 14-21 (v2)',
#            official_repo=None,
#            code_status='NO_PUBLIC_CODE_IDENTIFIED',
#            implementation_status='PAPER_FAITHFUL',
#            deviations=['Eq.19 gate convention printed as written; see docs'])
```

## 5. The four departures from blueprint §15

**(a) `targets/` is a new top-level module.** Blueprint §29 warns "do not mix
target definitions silently", but gives the rule no home. Zhao et al. and Wei et
al. use *different* 5-day realized-volatility estimators (see coverage matrix,
issue 6). Making each one a named, documented, unit-tested function is the only
way to keep reproductions honest.

**(b) `datasets/` ships real data.** The requirements forbid simulated data in
teaching examples. Fetching live at example-run time makes the docs
non-reproducible and breaks offline. Proposal: commit compressed daily CSVs for
the exact series the Tier-S papers use — S&P 500, DJI, NASDAQ, EUR/USD, gold,
Bitcoin — plus a `fetch_*` path via `yfinance` for users who want fresh data.
See open question 1.

**(c) `reporting/` is new.** "Publication-quality tables and figures" and
"reproduce equivalent tables and figures as closely as possible to the paper"
are hard requirements. That needs a real module: a journal style sheet,
LaTeX/HTML/Markdown table writers with significance stars and standard errors in
parentheses, and figure builders that mirror each paper's exhibits (Zhao Fig. 2
VaR violations; Wei Tables 2–3; RECH volatility-path panels; GINN λ parametric
study). Blueprint §15's `visualization/` is subsumed here.

**(d) `mean/` is split out of the models.** Blueprint §21 requires it; giving it
a module prevents every model re-implementing `eps_t = r_t − µ_t`.

## 6. Build order

Phases follow the blueprint roadmap; each phase ends with its tests green and
its docs page written before the next begins.

| Phase | Content | Gate |
|---|---|---|
| **1** | `core`, `constraints`, `distributions`, `losses`, `mean`, `targets`, `GARCHCell`/`GJRCell`/`EGARCHCell`, `forecast`, `risk`, `evaluation`, `reporting`, `datasets` | GARCH(1,1) recursion matches `arch` to 1e-10; constraint & gradient tests pass |
| **2** | `GARCHNN` family + `StructuralGARCHLSTM` (Zhao), `EmbeddedGARCHGRU`/`LSTM` (Wei) | Zhao Tables 2–5 and Wei Tables 2–3 reproduced qualitatively; recurrence unit tests exact |
| **3** | `NeuralGARCH` + variational engine (Yin), `RECH` + SMC (Nguyen) | RECH validated against the official `.mat` results we hold |
| **4** | `SigmaLSTM`, `SigmaCell`, `GARCHNet`, `GINN`/`GINN-0` | GINN λ-study reproduced |
| **5** | `RealRECH`/`DeepRGARCH`, realized-measure infrastructure | blocked on missing code + data — see `03_MISSING_RESOURCES.md` |
| **6** | Multivariate: `BEKK`, `NeuralBEKK`, `DCC`, `NeuralDCC`, `DCCGARCHMANN` | every `H_t` passes `eigvals > 0` |
| **7** | Feature hybrids & ensembles (Kim & Won, MIDAS-LSTM, GARCH-TFT, multi-GARCH Transformer, stacked ML-GARCH, XGBoost) | comparison experiment of the four GARCH-LSTM *kinds* runs end to end |
| **8** | Frontier: `GARCHFIS`, `GARCHDDQNVaR`, regimes/breaks, panel, graph | — |

Phase 1 is the only phase whose design I would like signed off before writing,
because everything else inherits its abstractions.

## 7. Testing strategy (blueprint §44)

- **Recurrence tests** — hand-computed 5-step traces for every cell.
- **Gradient tests** — `torch.autograd.gradcheck` in float64 on small inputs, for
  `raw_omega`, `raw_alpha`, `raw_beta` and neural weights.
- **Constraint tests** — 10 000 random raw draws ⇒ `ω>0, α≥0, β≥0, α+β<1`.
- **PSD tests** — every generated `H_t` and `R_t`.
- **Baseline tests** — our GARCH(1,1) filter vs `arch` to 1e-10; vs `tsgarch`
  reference values where cross-language checking is possible.
- **Paper-reproduction tests** — `test_zhao2024.py`, `test_wei2025.py`,
  `test_yin2022.py`, `test_rech2022.py`, `test_ginn2024.py`, … marked `slow`,
  asserting published numbers within a documented tolerance band.
- **Leakage tests** — an explicit test that no fitted scaler, GARCH filter or
  realized-volatility target ever sees post-split data.
