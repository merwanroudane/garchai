# Model Coverage Matrix — paper ↔ model ↔ code ↔ build decision

Phase 0 deliverable required by master blueprint §55 ("create model overlap
matrix, decide adapter vs reimplementation").

**Build decision legend**

- `IMPLEMENT` — write from paper equations; no usable prior implementation.
- `IMPLEMENT+VALIDATE` — write from paper equations, then numerically validate
  against official code we hold.
- `REPRODUCE` — write from paper; community code consulted for hyper-parameters
  only.
- `ADAPT` — wrap an existing mature package (`arch`, `mvgarch`); do not rebuild.

**Provenance uses the enum in master blueprint §14.**

---

## Tier S — structural neural-GARCH (the library's identity)

| # | Model class(es) | Paper | Code we hold | Provenance | Decision | Blockers |
|---|---|---|---|---|---|---|
| 1 | `GARCHNN`, `GARCHNNKernel`, `GJRNNKernel`, `FIGARCHNNKernel`, `StructuralGARCHLSTM` | Zhao et al., AAAI 2024 | none | `NO_PUBLIC_CODE_IDENTIFIED` | ✅ **BUILT** | Eq. 7 verified: `w=0` collapses to the kernel exactly (0.0 diff). Backprop recovers GARCH parameters to 2-3 dp vs `arch`. |
| 2 | `EmbeddedGARCHGRU`, `EmbeddedGARCHLSTM` (+ cells) | Wei, Yang & Cui 2025 (v2) | none | `NO_PUBLIC_CODE_IDENTIFIED` | ✅ **BUILT** | Both gate conventions available; econometric parameters verified interpretable after joint training. |
| 3 | `NeuralGARCH`, `NeuralBEKK` | Yin & Barucca 2022 | `neural-garch-main` — **not faithful**, unused | `INDEPENDENT_RELATED` | ✅ **BUILT** | VRNN + ELBO; time-varying (ω_t, α_t, β_t, ν_t). Eq. 4 sign typo corrected and recorded. |
| 4 | `RECH` (`SRNGARCH`, `SRNGJR`), `LSTMtGARCH` | Nguyen, Tran & Kohn, JAE 2022 | **official MATLAB, complete** | `OFFICIAL` | **IMPLEMENT+VALIDATE** | none — `.mat` result files let us check numbers |
| 5 | `RealRECH` / `DeepRGARCH`, `NeuralRealizedGARCH` | Liu, Wang, Tran & Kohn 2023 | ⚠️ **only RECH-SMC code; no DeepRGARCH** | `OFFICIAL` (incomplete) | **IMPLEMENT** | ⚠️ **code gap + data gap** (see `03_MISSING_RESOURCES.md`) |
| 6 | `SigmaLSTM`, `SigmaLSTMCell` | Rodikov & Antulov-Fantulin 2022 | none | `NO_PUBLIC_CODE_IDENTIFIED` | **IMPLEMENT** | ⚠️ 2 equation ambiguities (below) |
| 7 | `SigmaCell`, `SigmaRNN` | Rodikov & Antulov-Fantulin 2023 | none | `NO_PUBLIC_CODE_IDENTIFIED` | **IMPLEMENT** | deep read pending |
| 8 | `GARCHNet` | Buczyński & Chlebus 2023 | `garchnet-replication-main` | `COMMUNITY_REPLICATION` | **REPRODUCE** | official code is "available from author" only |
| 9 | `GINN`, `GINN0`, `GARCHInformedLoss` | Xu et al. 2024 | `GLoVE-main` (inspired, not official) | `COMMUNITY` | **REPRODUCE** | none — loss is Eq 17, unambiguous |

## Tier A — multivariate, graph

| # | Model class(es) | Paper | Code we hold | Decision | Blockers |
|---|---|---|---|---|---|
| 10 | `NeuralBEKK` (diagonal) | Yin & Barucca 2022 §2.2, Eqs 11–14 | none | ✅ **BUILT** | PSD verified at every step for n = 2..5, including a 50σ shock; asymmetry < 1e-12. |
| 11 | `NeuralDCC` | *no dedicated paper in folder* | none | **IMPLEMENT** (own extension, label `EXPERIMENTAL`) | design-only; no paper to be faithful to |
| 12 | `DCCGARCHMANN` | Fatima & Uddin 2022 | none | **REPRODUCE** — ✅ built. Both directions (`dcc_to_mann`, `mann_to_dcc`); own two-stage DCC (Engle 2002) since `arch` has no multivariate model. KSE-100/KLSE/BSESN unavailable → substituted per data rule. |
| 13 | `GENSHIN`, `MultiScaleGARCHGraphNet` | Yu, Xiao, Zhu & Zhang, *Finance Res. Lett.* 78:107196, 2025 | none | **IMPLEMENT** | ✅ unblocked. Needs VMD + a `csGARCH` we must build (`arch` has no component model). Chinese 5-min RV → substituted per data rule. |

## Tier B — feature hybrids, ensembles, transformers (baselines)

| # | Model class(es) | Paper | Code we hold | Decision |
|---|---|---|---|---|
| 14 | `FeatureGARCHLSTM` | Kim & Won 2018 | `LSTM_GARCH-1-master`, `GARCH-LSTM-main` | **REPRODUCE** (baseline) |
| 15 | `FeatureGARCHGRU`, `DLGARCH` | Michańków et al. 2023 | none | **REPRODUCE** |
| 16 | `LSTMANNGARCH` | Hu, Ni & Wen 2020 | none | **REPRODUCE** |
| 17 | `GARCHMIDASLSTM` | Ersin & Bildirici 2023 | none | **REPRODUCE** |
| 18 | `GARCHTFT` | Petrosino et al. 2025 | `garch-tft-main` (partial, **no licence** → nothing ported) | **REPRODUCE** — ✅ built from paper + Lim et al. (2021). Full TFT: GRN/GLU, variable selection, interpretable shared-`V` attention, pinball loss. |
| 19 | `MultiGARCHTransformer` | Gebreslase et al. 2026 | none | **REPRODUCE** |
| 20 | `StackedMLGARCH` | Rubio et al. 2026 | none | **REPRODUCE** |
| 21 | `GARCHXGBoost` / fusion | Maingo, Ravele & Sigauke 2025 | none | **REPRODUCE** |
| 22 | `SentimentGARCHX`, `SentimentGARCHLSTM` | Comput. Econ. 2026, 67:313–353 | none | **REPRODUCE** — ✅ built. Sentiment feed is proprietary; the series is a **required argument** (no invented data) and `vix_sentiment_proxy()` / the example's down-day proxy are labelled substitutes. |

## Tier C — frontier

| # | Model class(es) | Paper | Decision |
|---|---|---|---|
| 23 | `GARCHFIS` | Li & Zhang 2026 | **IMPLEMENT** — ✅ built. Fixed rule base; only antecedent widths adapt, scaled by the GARCH forecast. `kappa=0` is the ablation. |
| 24 | `GARCHDDQNVaR` | Pokou, Kamdem & Benhmidene, Comput. Econ. 2025 | **IMPLEMENT** — ✅ built. Reward per Eq. 12; the paper's prose and Eq. 12 disagree on the false-negative penalty, so both are reachable (`severe_false_negative`). |
| 25 | `ANNGARCH`, `LikelihoodGARCHLSTM` | Liu & So, *Information* 11(10):489, 2020 | **IMPLEMENT** — ✅ unblocked. `h_t = softplus(D(·))²`, standardized Student-t NLL, 20-day lookback, variable-importance by Δ log-likelihood. Bloomberg exog → substituted per data rule. |
| 26 | `NeuralGARCHEstimator`, `AmortizedGARCHInference` | De Clerk & Savel'ev, *Physica A* 603:127869, 2022 | **IMPLEMENT** — ✅ unblocked. Moment-map MLP (γ̂₁₅, Γ₄) → α₁, then β₁ from Eq 3, α₀ from Eq 4, plus the GARCH-feasibility guard. Open access (CC BY-NC-ND). |

## Classical econometric layer

| Family | Decision | Rationale |
|---|---|---|
| GARCH, GJR, EGARCH, APARCH, TGARCH, FIGARCH, RealizedGARCH | **ADAPT `arch`** for estimation baselines, **plus own differentiable cells** for the AI models | Blueprint §53: do not rebuild `arch`. But the cells must be torch-native and autograd-safe, which `arch` cannot provide. |
| GARCH-MIDAS | **IMPLEMENT** (thin) | not in `arch` |
| BEKK, DCC, ADCC | **IMPLEMENT** (torch, Cholesky-parameterised) + **ADAPT `mvgarch`** for baseline | PSD guarantees required |
| `tsgarch` / `rugarch` | **reference only** | cross-language numerical validation |

---

## Resolved-at-implementation issues found during the paper reads

These are recorded now so nothing is invented later (requirements line 67).

**1. Wei et al. 2025, Eq (19) — non-standard GRU convention.**
The paper's standard-GRU recap (Eq 13) is `h_t = (1−z_t)⊙h_{t−1} + z_t⊙h̃_t`, but
the GARCH-GRU cell (Eq 19) writes `ĥ_t = (1−z_t)⊙h̃_t + z_t⊙h_{t−1}` — the roles
are swapped. Faithful implementation follows **Eq 19 as printed**; the standard
convention will be available as `gate_convention="standard"` and the deviation
recorded in the model's `deviations` metadata.

**2. Yin & Barucca 2022, Eq (4) — sign typo in the Student-t log-likelihood.**
Printed as `… + logΓ(ν/2) + …` inside a term being maximised; the correct
Student-t log-density has `− logΓ(ν/2)`. Implement the mathematically correct
form, record the deviation.

**3. Rodikov & Antulov-Fantulin 2022, Eqs (8) and (11) — two ambiguities.**
`o_t = N(0, W_o[C_t²])` does not specify whether `W_o C_t²` is the variance or
the standard deviation, nor how the sample is reparameterised for gradients.
`σ̂²_t = ⟨C_t⟩²` uses an unglossed mean operator over the cell-state vector.
Eq (12), `L = Σ[−ln σ̂² − r²/σ̂²]`, is written as a quantity to be *maximised*
(it is twice the Gaussian log-likelihood kernel) though labelled a loss.
→ The σ-Cell paper (2309.01565, 66 pp) is the resolution route; it will be
deep-read before `SigmaLSTM` is written. If it does not resolve them, this
becomes a question for you rather than a guess.

**4. RECH — paper vs official code differ, by the authors' own statement.**
Repo README: *"in the JAE paper, we implemented RNN-GARCH for the RECH
specification. The LSTM-tGARCH specification is more sophisticated and works
better."* The paper specifies SRN-GARCH with `ε_t ~ N(0,1)` (Eqs 8a–8d); the
shipped code implements LSTM-tGARCH with Student-t errors and SMC inference.
→ This is exactly the case master blueprint §64 anticipates. Both are built and
selected by `implementation="paper"` / `implementation="official_code"`.

**5. Zhao et al. 2024, Eq (6) — T-loss is a kernel, not a full log-density.**
The printed T-loss omits the `logΓ` normalising constants. Since they do not
depend on `σ̂_t` they do not change the gradient w.r.t. the variance path, but
they *do* change reported log-likelihood values. Implement both: `tloss_kernel`
(paper-faithful, for reproduction) and `student_t_nll` (complete, for model
comparison and AIC/BIC).

**6. Zhao et al. and Wei et al. use different realized-volatility targets.**
Zhao: `σ_t = sqrt(Σ_{i=0}^{k−1} ε²_{t−i})`, k=5, **not** divided by k, **not**
demeaned. Wei: `σ_t = sqrt((1/k)Σ_{i=1}^{k}(r_{t−i} − µ_k)²)`, k=5, demeaned, and
**lagged** (uses `t−1…t−k`, so no look-ahead). These are different estimators and
must not be shared. Both go in `garchai.targets` as named, documented functions.
