# Reproduction log

Empirical results from running the implemented models against the protocols
their papers specify. Recorded whether or not they agree with the published
findings — a reproduction attempt that only gets written up when it succeeds is
not a reproduction attempt.

---

## Xu et al. (2024), GINN — **partially reproduced**

**Protocol.** The paper's own: rolling 90-day window, AR mean, GARCH(1,1)
refitted at *every* step (`refit_garch_every=1`), ground-truth variance
`(r_t − µ̂_t)²`, chronological 70/30 split.

**Data.** S&P 500 daily log returns, 2016-01-01 to 2019-12-31 (825 usable
windows, 247 test). The paper uses seven indices over 1992–2022.

| Model | R² | MSE | MAE |
|---|---|---|---|
| GARCH baseline | −0.369 | 2.044 | 0.811 |
| GINN-0 (λ = 0) | −0.119 | 1.621 | 0.715 |
| GINN (λ = 0.01) | −0.058 | 1.527 | 0.694 |
| λ = 1 (data only) | **+0.038** | **1.384** | **0.610** |

**What reproduces.** Both GINN and GINN-0 beat the GARCH baseline decisively —
25% lower MSE — and GINN ≈ GINN-0, which is the paper's headline surprise. Both
of those hold.

**What does not.** The paper's parametric study selects λ = 0.01 as optimal.
On this sample the *data-only* network (λ = 1) is best on all three metrics.
So the specific claim that a near-zero λ is optimal does **not** reproduce here.

**Why the discrepancy is plausible, and why it is not being explained away.**
Three differences could account for it: a much shorter and calmer sample
(2016–2019 versus 1992–2022, which excludes both 2008 and 2020); a smaller
network (2 layers of 64 versus 3 of 256); and fewer epochs (60 versus 300).
The paper also tuned λ on NASDAQ specifically and applied it to the other six
indices, so λ = 0.01 was never claimed to be universally optimal. None of that
is verified, and none of it is offered as a resolution — the honest statement is
that on this sample and configuration, the optimal λ is not 0.01.

**A methodological note worth keeping.** An earlier run used
`refit_garch_every=10` to save time. Under that shortcut λ = 1 was the *worst*
model rather than the best, which would have led to the opposite conclusion.
The rolling-GARCH refit frequency is not a performance knob; it changes the
answer.

---

## Zhao et al. (2024), GARCH-NN — **structural claims verified**

`StructuralGARCHLSTM(w=0)` reproduces its GARCH kernel to `0.0e+00`, confirming
Eq. 7's design intent that the LSTM modulates a valid variance rather than
replacing it.

`GARCHNN` estimated by backpropagation recovers a known GARCH(1,1):

| | ω | α | β |
|---|---|---|---|
| True | 0.0500 | 0.0900 | 0.8800 |
| `GARCHNN` (backprop) | 0.0289 | 0.0755 | 0.9073 |
| `arch` (MLE) | 0.0302 | 0.0759 | 0.9055 |

Agreement with `arch` to 2–3 decimals, which is the paper's Table 2 claim. Both
estimators miss the true ω in the same direction and by the same amount, so the
gap is sampling error at n = 1500, not an estimator defect.

---

## Nguyen et al. (2022), RECH — **structural claims verified**

`RECH(beta_1=0)` collapses to plain GARCH to `4.4e-16`. Theorem 1's requirement
that `ω_t ≤ M` holds exactly: with `activation_bound=1.0` the fitted `ω_t` path
lies in `[β₀, β₀+β₁]`, which is what the bounded ReLU is there to guarantee.

## Liu et al. (2023), DeepRGARCH — **structural claims verified**

`DeepRGARCH(beta_1=0)` reproduces `RealizedGARCH` to `0.0e+00`. The measurement
equation is confirmed live: zeroing φ moves the joint negative log-likelihood by
208, so ξ, φ, τ₁, τ₂ and σ_u are genuinely identified rather than inert.

Not yet attempted against the paper's own numbers: the Oxford-Man Realized
Library is discontinued, and no reference implementation of DeepRGARCH exists to
check the SMC configuration against.


---

## Feature hybrids and ensembles — S&P 500, 2015-2019

Rolling 400-day GARCH/GJR features, 5-day rolling-standard-deviation target,
chronological 80/20 split, 170 test observations.

| Model | MAE | RMSE | R² |
|---|---|---|---|
| BLSTM-ANN-GARCH | 0.0689 | 0.0873 | +0.965 |
| Multi-GARCH Transformer | 0.0649 | 0.0858 | +0.964 |
| Stacked ML-GARCH | 0.0663 | 0.1057 | +0.938 |
| LSTM-ANN-GARCH | 0.1927 | 0.2401 | +0.687 |
| GARCH-XGBoost | 0.2430 | 0.3322 | +0.635 |
| *GARCH baseline* | — | 0.5445 | — |

**Read the R² with care.** These are far higher than the GINN numbers above, and
the reason is the *target*, not the models: a 5-day rolling standard deviation is
strongly autocorrelated and partly recoverable from the lagged returns already in
the feature set, whereas GINN's target is a single squared residual. The two
tables are not comparable, and quoting the 0.96 as evidence these models are
"better" than GINN would be wrong.

**GARCH-XGBoost works as designed.** RMSE 0.3322 against a GARCH baseline of
0.5445 — a 39% reduction — and because the booster is fitted to the *residual*,
that gain is orthogonal to what GARCH already knew rather than a relabelling of
it.

**Stacked ML-GARCH did not beat its own best base learner**: RMSE 0.1057 against
0.0569 for the best of ridge / random forest / SVR. The class reports
``beats_best_base: False`` rather than quoting only its own error. That is the
expected hazard of stacking on a short sample — the meta-learner has 682 rows
and five base-model columns to fit — and it is the reason the metric exists.
Whether the paper's result holds on its own Bitcoin sample is untested here.

---

## 6. The frontier family (GARCH-FIS, DDQN VaR, DCC-MANN, GARCH-TFT, sentiment)

Real data throughout: `examples/07_frontier_models.py`. Each model is reported
next to the ablation that isolates the paper's actual contribution, because for
three of these five the pipeline works and the *mechanism* is what is in doubt.

### 6.1 GARCH-FIS — the adaptation does not replicate on returns

S&P 500 daily returns, 2015-2020, chronological 80/20 split. Both rows are the
same six-rule Takagi-Sugeno system; they differ only in whether the membership
widths scale with the GARCH forecast.

| Model | MAE | RMSE | R² |
|---|---|---|---|
| FIS, no adaptation (κ = 0) | 1.2030 | **2.0038** | **+0.0096** |
| GARCH-FIS (κ = 1) | **1.1894** | 2.0311 | −0.0151 |

**This does not reproduce the paper's claim.** The volatility adaptation is the
paper's sole contribution, and switching it on improves MAE while making both
RMSE and R² worse. Splitting the difference: κ = 1 wins on the metric that
weights all errors equally and loses on the one that punishes large errors, which
is consistent with wider membership functions smoothing the fit in calm periods
at the cost of the tails — the opposite of the intended effect.

Two caveats, neither of which rescues the result:

1. **The target differs from the paper's.** The paper forecasts *prices*, which
   are near-unit-root and strongly autocorrelated; both R² values here are
   within noise of zero because daily *returns* are close to unpredictable. A
   price-level replication would show large R² for both rows and might separate
   them differently.
2. **The width-scaling law is our own.** The paper states that width and universe
   coverage adapt, but gives no closed form (see `deviations`). A different
   scaling could behave differently. This is recorded as an untested conjecture,
   not an explanation.

### 6.2 GARCH-DDQN VaR — reproduces, and for the stated reason

FTSE 100, 2015-2020, α = 5%, 25 episodes.

| Model | Violation rate | \|rate − 0.05\| | Kupiec p | Christoffersen p | Recall |
|---|---|---|---|---|---|
| GARCH-DDQN | 0.0503 | 0.0003 | 0.9577 | **0.5513** | 0.605 |
| Unconditional quantile | 0.0501 | 0.0001 | 0.9812 | 0.1275 | — |

Both models hit the nominal rate almost exactly, so Kupiec cannot separate them —
which is the point. The separation is in **Christoffersen** (0.55 vs 0.13): the
unconditional threshold gets the long-run frequency right while bunching its
violations in turbulent stretches, and the conditional model does not. That is
precisely the paper's claim ("reduces both the frequency and the temporal
clustering of VaR violations"), and it is visible only because the unconditional
benchmark is reported alongside.

**A bug found and fixed en route.** The ε-greedy schedule initially decayed at
0.995 *per episode* — the standard per-*step* value — leaving the agent 86%
random after the default 30 episodes. Training reward went *down* (−494.9 →
−519.1). The learned policy was still decent because DQN is off-policy and the Q
function trains from replay regardless of behaviour, which is exactly what made
the bug easy to miss: only the learning curve was wrong. After switching to an
anneal-to-floor schedule, reward runs −494.9 → **+70.5** and recall rises
0.47 → 0.60. Regression test: `test_epsilon_anneals_to_its_floor`.

### 6.3 DCC-GARCH-MANN — both directions, with the ordering mattering

Four-market panel (S&P 500, FTSE 100, Nikkei 225, Hang Seng), 2015-2020.
**Data substitution:** the paper uses KSE-100, KLSE and BSESN, none of which are
in the bundled public sources; Nikkei and Hang Seng stand in, so these numbers
are *not* comparable to the paper's Table 5.

| Model | RMSE | MAE |
|---|---|---|
| DCC-GARCH → MANN | **0.9896** | **0.7160** |
| MANN → DCC-GARCH | 1.0494 | 0.7501 |

Feeding the econometric second moments *into* the network beats fitting DCC to
the network's residuals, which is the direction the paper's title emphasises.

Estimator validation on simulated data, where the truth is known:

- Known constant correlation 0.60 → recovered 0.607; every `R_t` has unit
  diagonal and is positive definite.
- Known DCC process (a = 0.05, b = 0.90) → recovered a = 0.0546, b = 0.8771.
- Constant-correlation data → a + b ≈ 0.056, i.e. the estimator correctly
  reports *no* correlation dynamics rather than manufacturing them.

### 6.4 GARCH-TFT — directionally consistent with the paper

S&P 500 with full OHLC, 2015-2020, chronological 80/20 split. Each proxy is run
twice: with the rolling GARCH forecast among the inputs, and without it (the
paper's stand-alone TFT).

| Target | Model | MAE | RMSE | R² | GARCH weight | crossing rate |
|---|---|---|---|---|---|---|
| HV | GARCH-TFT | **0.0529** | **0.1469** | 0.6048 | 0.187 | 0.000 |
| HV | TFT, no GARCH | 0.0695 | 0.1593 | 0.5957 | — | 0.242 |
| GK | GARCH-TFT | **0.0439** | **0.0782** | **0.6632** | 0.160 | 0.000 |
| GK | TFT, no GARCH | 0.0452 | 0.0841 | 0.6059 | — | 0.000 |

The paper's claim is *proxy-specific*: a significant gain on Garman-Klass, mere
parity on historical volatility. That direction reproduces. The R² gain from
adding GARCH is **+0.057 on GK against +0.009 on HV** — a clear improvement on
the range-based proxy and near-parity on the close-to-close one, which is the
pattern the paper reports. The magnitudes are not comparable (the paper uses
ETFs, we use the index), so only the ordering is being checked here.

**The model does use its GARCH input, but sparingly.** The variable-selection
weight on the GARCH column is 0.16-0.19 against the 0.33 a uniform allocation
would give, so the network down-weights it relative to the lagged proxy while
still extracting something from it. That number is reported precisely because
an error-metric gain alone cannot distinguish "the GARCH term helped" from
"the extra column happened to help training".

**An incidental finding worth more than the headline.** The no-GARCH TFT on HV
crossed 24% of its quantile pairs, which makes its 0.78 coverage figure
meaningless — a crossed interval is empty. Adding the GARCH input took crossing
to zero. We do not have a tested explanation for this and offer none; it is
recorded because a coverage number quoted without its crossing rate would have
hidden it entirely.

### 6.5 Sentiment-augmented GARCH — implemented, not reproduced

**Blocked on data, and left blocked.** The paper's sentiment feed over S&P 500
constituents is proprietary and not redistributable. The sentiment series is
therefore a **required argument**: the module will not invent one. Two documented
public substitutes are provided (`vix_sentiment_proxy`, and a down-day proxy in
the example), and every figure computed from them is labelled a proxy result.
**No claim is made that the paper's results reproduce**, because the input that
carries its central variable is absent.

Parameter recovery on a known DGP (ω = 0.02, α = 0.08, β = 0.85, γ = 0.15):
α → 0.0797, β → 0.8638, γ → 0.0956. α and β are recovered accurately; γ is
underestimated by roughly a third at n = 600, which is worth knowing before
reading any γ as an effect size.

**The public proxy adds nothing, and the model says so.** On S&P 500
2015-2020 with a down-day proxy, the optimiser drives γ to 0, the log-likelihood
is identical to plain GARCH to two decimals (−458.12), the likelihood-ratio test
of γ = 0 gives p = 1.000, and `sentiment_contribution` is 0.000. This is the
expected outcome, not a failure: the proxy is a deterministic function of past
returns, so it carries no information GARCH does not already have. It does
confirm that the augmentation reports itself as inert rather than manufacturing
an effect — which is the property that matters when a *real* sentiment feed is
substituted in.

Positivity is structural rather than patched: γ ≥ 0 by an exponential
parameterisation and S ∈ [0, 1] by `normalise_sentiment`, so the sentiment term
cannot drive the variance negative. The paper does not say how it handles this.

### 6.6 Two presentation bugs fixed in the tables themselves

Worth recording because they would have produced misleading *journal* output
from correct code:

1. **`comparison_table` bolds the best value per column**, which silently
   asserted that a higher `mean_VaR`, a higher `dcc_persistence` and a higher
   Kupiec p-value were *better*. None of the three is a score. Those tables are
   now built directly, with descriptive parameters split into their own
   unranked table and violation accuracy reported as |rate − nominal|.
2. **Quantile crossing in GARCH-TFT.** Nothing constrains q₁₀ ≤ q₉₀ — the
   pinball loss is separable across quantiles — and roughly 9% of predicted
   pairs crossed on one run, which makes the interval empty and any coverage
   figure computed from it meaningless. Rather than silently sorting the
   outputs (Lim et al.'s TFT does not enforce monotonicity either),
   `quantile_crossing_rate` is reported alongside `interval_coverage` so the
   coverage number cannot be quoted without its own validity check.

Also noted: in-sample interval coverage is not evidence. On one run it fell from
0.99 to 0.29 as training continued, because an overfit model narrows its band
around values it has memorised. Coverage is only meaningful out of sample, where
the same configuration gave 0.70 against a nominal 0.80.


### 6.7 A silent-failure bug in the sentiment estimator

Worth its own entry because the failure mode was invisible. The public proxy's
first rolling window is empty, and the mean of an empty slice is `NaN`. The
non-negativity guard read `if s.min() < 0`, and **`nan < 0` is `False`**, so the
NaN passed validation, made every likelihood evaluation `NaN`, and Nelder-Mead
returned the infeasibility penalty with the starting values untouched. The
published table showed `loglik = -1e10` beside α = 0.0800 and β = 0.8800 — which
are exactly `theta0`. A reader skimming the α and β columns would have seen
plausible GARCH parameters and no sign that nothing had been estimated.

Three fixes, in order of importance:

1. Finiteness is now checked **before** the sign test, with a message naming the
   rolling-window cause.
2. The estimator raises if the optimiser ends on the penalty value, rather than
   reporting a total failure as an estimate.
3. The example seeds its first window at neutral (0.5) instead of `NaN`.

Regression tests: `test_sentiment_rejects_nan_before_the_sign_check`,
`test_sentiment_rejects_non_finite_returns`.

### 6.8 A dtype bug that only appears outside the test suite

Every test module sets `torch.set_default_dtype(torch.float64)` at import. The
package documents float64 as its numerical default but does not set it globally,
so models that called `torch.as_tensor` on a float64 array without naming a
dtype built `Double` inputs for `Float` layers — passing every test and failing
the first time a user ran an example script (`mat1 and mat2 must have the same
dtype`). The three new torch models now convert with
`torch.get_default_dtype()` explicitly, so they work under either default.
Regression test: `test_models_work_under_float32_default`, which flips the
global dtype to float32 and restores it in a `finally`.

### 6.9 Test-suite runtime

The suite was taking over four hours at one point. That figure was an artifact —
it was measured while the frontier example ran concurrently and the two
processes fought for CPU. Measured alone it was **12m30s**, of which a single
pre-existing test, `test_garch_nn_recovers_a_known_garch_process`, accounted for
355s (47%).

The genuine cost in the new models was `GARCHFIS._volatility_path` and
`GARCHDDQNVaR._garch_sigma`, which re-estimated GARCH by full maximum
likelihood at **every** step: with 700 observations and a 250-day window that is
~450 optimisations per call, and `fit`, `predict`, `var` and `evaluate` each
paid it again. Two changes, both following the convention already established by
`build_garch_features`:

- **`refit_every`** (default 1, the exact protocol) re-estimates every k-th step
  and filters the variance recursion forward in between. It never touches future
  data, so it does not leak, but the parameters lag by up to k observations and
  it is therefore an approximation. `GARCHFIS.uses_exact_protocol` reports which
  one produced a given result — the GINN reproduction in this package flipped
  its conclusion between `refit_every=10` and `refit_every=1`, so this is not a
  cosmetic distinction.
- **Caching** the volatility path on the DDQN, since `fit`/`var`/`evaluate` are
  normally called on the same series.

The twelve tests over ~10s now carry the existing `slow` marker, giving:

| Command | Tests | Time |
|---|---|---|
| `pytest -m "not slow"` | 539 | **1m46s** |
| `pytest --doctest-modules src/garchai` (full + doctests) | 597 | **5m43s** |

The slow tests are not skippable in CI — they are the parameter-recovery and
training-convergence checks, which are the ones that would catch a real
regression — but the marker gives a fast local loop.
