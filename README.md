# garchai

**Econometrically constrained, differentiable, AI-enhanced conditional heteroskedasticity models.**

`garchai` implements **24 published GARCH-AI papers** as a single, consistent Python library. Every model is built from its source paper, carries machine-readable provenance, and states its own deviations from the paper out loud.

- **PyTorch-native** — every variance recursion is differentiable and autograd-safe
- **Validated against `arch`** to 4.4e-15 on the full conditional-variance path
- **Honest by construction** — models report when they fail to beat their own baseline
- **597 tests**, MIT licensed

```bash
pip install garchai
```

---

## Table of contents

1. [Is this the right library for you?](#1-is-this-the-right-library-for-you)
2. [Installation](#2-installation)
3. [Five-minute quickstart](#3-five-minute-quickstart)
4. [The four concepts you need](#4-the-four-concepts-you-need)
5. [Tutorial 1 — Your first GARCH model](#tutorial-1--your-first-garch-model)
6. [Tutorial 2 — Value-at-Risk and backtesting](#tutorial-2--value-at-risk-and-backtesting)
7. [Tutorial 3 — Your first neural volatility model](#tutorial-3--your-first-neural-volatility-model)
8. [Tutorial 4 — Feature hybrids (GARCH → machine learning)](#tutorial-4--feature-hybrids-garch--machine-learning)
9. [Tutorial 5 — Comparing models properly](#tutorial-5--comparing-models-properly)
10. [Tutorial 6 — Publication tables and figures](#tutorial-6--publication-tables-and-figures)
11. [Complete model catalogue](#5-complete-model-catalogue)
12. [Recipes](#6-recipes)
13. [Running the tests](#7-running-the-tests)
14. [Troubleshooting](#8-troubleshooting)
15. [Provenance and honesty](#9-provenance-and-honesty)
16. [Citation](#10-citation)

---

## 1. Is this the right library for you?

**Use `garchai` if you want to:**

- fit a neural-network volatility model that still obeys econometric constraints
- reproduce a specific GARCH-AI paper without re-deriving it from the PDF
- compare 20+ hybrid architectures on the same data with the same evaluation
- produce journal-format tables and figures for a paper

**Use something else if you want:**

| You want | Use instead |
|---|---|
| Plain GARCH/EGARCH/GJR estimation only | [`arch`](https://github.com/bashtage/arch) — mature, faster, and `garchai` validates against it |
| Plain GARCH-LSTM / GARCH-GRU feature hybrids | `hybridecon` — `garchai` deliberately does not duplicate it |
| R workflows | [`tsgarch`](https://cran.r-project.org/package=tsgarch) |

`garchai` builds **what those libraries do not cover**: models where the neural network is *inside* the variance recursion, informed losses, graph and decomposition pipelines, multivariate hybrids, fuzzy and reinforcement-learning approaches.

### New to volatility modelling entirely?

You need three ideas, and nothing more, to use this library:

1. **Returns** are percentage price changes. They are nearly unpredictable in *level* but very predictable in *magnitude*.
2. **Volatility clusters** — big moves follow big moves. GARCH models this: today's variance depends on yesterday's shock and yesterday's variance.
3. **Conditional variance** `h_t` is what every model here predicts. Volatility is `sqrt(h_t)`.

That is enough. Start at [Tutorial 1](#tutorial-1--your-first-garch-model).

---

## 2. Installation

### Basic

```bash
pip install garchai
```

This gives you the core: all differentiable cells, distributions, structural neural models, risk tools, and the bundled datasets.

### With optional extras

Some models need extra packages. Install only what you use:

```bash
pip install "garchai[benchmarks]"
```

| Extra | Installs | Needed for |
|---|---|---|
| `benchmarks` | `arch`, `statsmodels`, `scikit-learn`, `xgboost` | rolling-GARCH features, `GARCHXGBoost`, `StackedMLGARCH`, `GARCHFIS`, `GARCHDDQNVaR`, `DCCGARCHMANN` |
| `plots` | `matplotlib` | every figure function |
| `data` | `yfinance`, `pandas-datareader` | downloading your own data |
| `tuning` | `optuna` | hyper-parameter search |
| `dev` | `pytest`, `ruff`, `mypy` | running the test suite |
| `all` | everything above | trying everything |

Most users should start with:

```bash
pip install "garchai[benchmarks,plots]"
```

### Verify the install

```python
import garchai
print(garchai.__version__)

from garchai.datasets import list_datasets
print(list_datasets())     # DataFrame of the 9 bundled series
```

---

## 3. Five-minute quickstart

Copy this whole block. It uses **real S&P 500 data bundled with the package** — no download needed.

```python
import warnings
warnings.filterwarnings("ignore")

import torch
torch.set_default_dtype(torch.float64)          # see "Concept 1" below — do this first

from garchai.datasets import load_returns
from garchai.econometric import GARCH

# 1. Load real data. Returns come back in PERCENT (mean |r| is about 0.7).
returns = load_returns("sp500", start="2015-01-01", end="2020-12-31")
print(returns.head())

# 2. Build and fit a model. Note: fit() returns a RESULT, not the model.
model = GARCH()
result = model.fit(returns)

# 3. Inspect what you got
print(model.summary())

# 4. Forecast the next 5 days' variance
forecast = model.forecast(horizon=5)
print("variance path:", forecast.variance)

# 5. Risk numbers
print("5% VaR, last 3 days:", model.var(alpha=0.05)[-3:])
```

Expected output (abridged):

```
  Observations    : 1510
  Log-likelihood  : -1841.9180
  AIC / BIC       : 3693.8359 / 3720.4353

  Variance parameters
    omega         0.041830
    alpha         0.237629
    beta          0.733207

  Persistence      : 0.970837
  Long-run variance: 1.43433
```

**Read the persistence number.** `alpha + beta = 0.97` means shocks to volatility decay slowly — a 1% surprise today still moves variance months later. That is the single most informative number a GARCH fit gives you.

---

## 4. The four concepts you need

Everything in this library rests on four things. Learn these and the rest is API surface.

### Concept 1 — Set float64 first

```python
import torch
torch.set_default_dtype(torch.float64)
```

**Do this before creating any model.** Likelihoods involve `log(h_t)` where `h_t` can be very small; in float32 the gradients are unreliable and fits silently degrade. The package documents float64 as its default but does not set it globally on import, because changing a global on import is rude to the rest of your program.

If you forget, you may see `mat1 and mat2 must have the same dtype`. That is this.

### Concept 2 — Returns must be in percent

```python
returns = load_returns("sp500")        # already percent — mean |r| is about 0.7
returns = 100 * np.diff(np.log(prices))  # if you build your own, multiply by 100
```

GARCH optimisers are badly conditioned on raw decimal returns (variance ~1e-4). Every bundled loader returns percent. If you pass decimals you will usually get a fit that "converges" to nonsense.

If you cannot change your data, pass `ModelConfig(scale="auto")` and the model will rescale internally and report the scale it used.

### Concept 3 — `fit()` returns a result, the model holds the state

```python
model = GARCH()
result = model.fit(returns)     # result: log-likelihood, AIC, paths, convergence
model.summary()                 # the MODEL is what you keep using
model.forecast(horizon=5)
model.var(0.05)
```

This trips people up. `result` is a `FitResult` — a record of the fit. `model` is the fitted object.

`FitResult` carries: `loglikelihood`, `aic`, `bic`, `converged`, `message`, `conditional_variance`, `conditional_volatility`, `long_run_variance`, `n_obs`, `n_params`, `history`.

### Concept 4 — Two config objects control everything

```python
from garchai.core import ModelConfig, TrainConfig

# WHAT the model is
config = ModelConfig(
    mean="constant",         # "zero" | "constant" | "ar" | "arma" | "neural"
    distribution="student-t",# "normal" | "student-t" | "skew-t" | "ged"
    scale="auto",            # rescale badly-scaled data and report it
    seed=0,
)

# HOW it is fitted
train = TrainConfig(
    optimizer="adam",        # "adam" | "adamw" | "lbfgs"
    lr=0.01,
    max_epochs=800,
    patience=20,
    refine_with_lbfgs=True,  # polish the Adam solution — usually worth it
)

model = GARCH(config=config)
model.fit(returns, train)
```

**If `summary()` says `Status: maximum epochs reached`, your fit did not converge.** Raise `max_epochs` and set `refine_with_lbfgs=True`. This is the most common cause of odd parameter estimates.

`TrainConfig` rejects `shuffle=True` — shuffling destroys the time ordering these models depend on.

---

## Tutorial 1 — Your first GARCH model

**Goal:** fit, diagnose, and forecast. Fifteen minutes.

### Step 1 — Load data

```python
import torch, warnings
warnings.filterwarnings("ignore")
torch.set_default_dtype(torch.float64)

from garchai.datasets import load_returns, list_datasets

print(list_datasets())     # a DataFrame: key, name, ticker, bundled, used_by
# keys: sp500, nasdaq, dji, ftse, nikkei, hangseng, bitcoin, gold, eurusd

returns = load_returns("sp500", start="2015-01-01", end="2020-12-31")
```

### Step 2 — Fit three specifications and compare

```python
from garchai.econometric import GARCH, GJRGARCH, EGARCH
from garchai.core import ModelConfig, TrainConfig

train = TrainConfig(max_epochs=800, refine_with_lbfgs=True)
config = ModelConfig(distribution="student-t")   # fat tails: almost always better

models, results = {}, {}
for name, cls in [("GARCH", GARCH), ("GJR", GJRGARCH), ("EGARCH", EGARCH)]:
    m = cls(config=config)
    results[name] = m.fit(returns, train)
    models[name] = m

for name, r in results.items():
    print(f"{name:7s}  loglik={r.loglikelihood:10.2f}  AIC={r.aic:9.2f}  converged={r.converged}")
```

**How to read this:** lower AIC is better. GJR and EGARCH usually beat plain GARCH on equity data because they capture the **leverage effect** — negative shocks raise volatility more than positive ones of the same size.

### Step 3 — See the leverage effect

```python
from garchai.reporting import plot_news_impact, use_journal_style
import matplotlib.pyplot as plt

use_journal_style()
fig = plot_news_impact({k: v for k, v in models.items()})
plt.savefig("news_impact.png", dpi=300)
```

The news-impact curve plots tomorrow's variance against today's shock. A symmetric GARCH gives a parabola; GJR and EGARCH give an asymmetric curve, steeper on the left.

### Step 4 — Forecast

```python
f = models["GJR"].forecast(horizon=10)
print(f.variance)          # variance path
print(f.to_frame())        # tidy DataFrame

# For EGARCH, multi-step forecasts have no closed form — the package
# simulates and tells you so:
f = models["EGARCH"].forecast(horizon=10, method="simulation", n_simulations=10000, seed=0)
print(f.method)            # "simulation"
```

### Step 5 — Check the provenance

```python
print(models["GARCH"].provenance)
```

Every model prints its source paper, the equations implemented, and any deviation. This is what makes the library citable.

---

## Tutorial 2 — Value-at-Risk and backtesting

**Goal:** produce a VaR series and test whether it is actually correct.

```python
import torch, warnings
warnings.filterwarnings("ignore"); torch.set_default_dtype(torch.float64)

from garchai.datasets import load_returns
from garchai.econometric import GARCH
from garchai.core import ModelConfig, TrainConfig
from garchai.risk import var_backtest_table

returns = load_returns("sp500", start="2015-01-01", end="2020-12-31")

model = GARCH(config=ModelConfig(distribution="student-t"))
model.fit(returns, TrainConfig(max_epochs=800, refine_with_lbfgs=True))

var = model.var(alpha=0.05)     # negative numbers: a return LEVEL
es  = model.es(alpha=0.05)      # expected shortfall

table = var_backtest_table(returns.to_numpy(), var, alpha=0.05, es=es)
print(table.to_markdown())
```

Output:

| test | statistic | p_value | violation_rate | expected_rate | reject_5pct |
|---|---|---|---|---|---|
| Kupiec POF (unconditional coverage) | 7.043 | 0.008 | 0.0656 | 0.05 | **True** |
| Christoffersen independence | 1.063 | 0.303 | 0.0656 | 0.05 | False |
| Christoffersen conditional coverage | 8.105 | 0.017 | 0.0656 | 0.05 | **True** |

### Read this failure — it is the lesson

The model is **rejected**. It produces 6.6% violations where 5% were promised. This is a real, expected result and worth understanding before you trust any VaR number:

- **In-sample VaR is not a backtest.** The parameters saw these same returns. Genuine evaluation needs a rolling out-of-sample scheme.
- **Kupiec** tests the *rate* only. **Christoffersen independence** tests whether violations *cluster*. Here independence passes (0.303) while coverage fails — the model gets the timing right and the level wrong.
- A VaR model can pass one and fail the other. Always report both, which is why `var_backtest_table` returns them together.

**Sign convention:** VaR and ES are returned as return *levels*, so they are negative. A violation is `returns < var`.

Individual tests are available too:

```python
from garchai.risk import (
    kupiec_pof, christoffersen_independence, christoffersen_conditional_coverage,
    dynamic_quantile_test, expected_shortfall_backtest, violation_ratio,
)

print(kupiec_pof(returns.to_numpy(), var, alpha=0.05))
print(dynamic_quantile_test(returns.to_numpy(), var, alpha=0.05, lags=4))
```

---

## Tutorial 3 — Your first neural volatility model

**Goal:** replace part of the GARCH recursion with a neural network, without losing the constraints that make it a volatility model.

The key idea: in `GARCHNN` (Zhao et al. 2024) the network output is *added* to a GARCH kernel. Set the network weight to zero and you recover GARCH exactly — the package tests this to 0.0e+00.

```python
import torch, warnings
warnings.filterwarnings("ignore"); torch.set_default_dtype(torch.float64)

from garchai.datasets import load_returns
from garchai.structural import GARCHNN
from garchai.core import ModelConfig, TrainConfig

returns = load_returns("sp500", start="2015-01-01", end="2020-12-31")

model = GARCHNN(
    kernel="garch",                              # "garch" | "gjr" | "figarch"
    config=ModelConfig(distribution="student-t"),
)
result = model.fit(returns, TrainConfig(lr=1e-3, max_epochs=300, patience=30))

print(model.summary())
print("converged:", result.converged)
```

### Compare against the model it generalises

A neural model that cannot beat plain GARCH is not worth its parameters. Always check:

```python
from garchai.econometric import GARCH

baseline = GARCH(config=ModelConfig(distribution="student-t"))
base_result = baseline.fit(returns, TrainConfig(max_epochs=800, refine_with_lbfgs=True))

print(f"GARCH   loglik={base_result.loglikelihood:.2f}  AIC={base_result.aic:.2f}")
print(f"GARCHNN loglik={result.loglikelihood:.2f}  AIC={result.aic:.2f}")
```

AIC penalises the network's extra parameters. If the neural model does not win on **AIC**, the extra flexibility is not paying for itself on your data.

### Other structural models, same interface

Every model below takes `returns` and follows the identical `fit / summary / forecast / var` pattern:

```python
from garchai.structural import (
    RECH,          # Nguyen et al. — recurrent conditional heteroskedasticity
    SigmaCell,     # Rodikov & Antulov-Fantulin — GARCH as an RNN cell
    GARCHNet,      # Buczynski & Chlebus — time-varying distribution shape
    NeuralGARCH,   # Yin & Barucca — variational, time-varying parameters
    ANNGARCH,      # Liu & So — likelihood-based ANN-GARCH
)

model = RECH(garch_component="gjr", implementation="paper")
model.fit(returns, TrainConfig(lr=1e-3, max_epochs=300))
```

---

## Tutorial 4 — Feature hybrids (GARCH → machine learning)

**Goal:** use GARCH forecasts as *inputs* to a learner. This is the most common hybrid design in the literature and the easiest place to accidentally cheat.

### The look-ahead trap

If you fit GARCH on the **whole** sample and feed its fitted variance to a model that predicts part of that sample, the GARCH parameters already saw the future. Accuracy looks superb and vanishes in production.

`build_garch_features` prevents this by fitting GARCH on a **rolling window** that only ever sees the past.

```python
import numpy as np, torch, warnings
warnings.filterwarnings("ignore"); torch.set_default_dtype(torch.float64)

from garchai.datasets import load_returns
from garchai.ensembles import build_garch_features, LSTMANNGARCH
from garchai.core import TrainConfig

returns = load_returns("sp500", start="2015-01-01", end="2020-12-31").to_numpy()

# Target: 5-day rolling standard deviation (a common volatility proxy)
target = np.array([returns[max(0, t-5):t].std() for t in range(returns.size)])
target[:5] = target[5]

features = build_garch_features(
    returns, target,
    specifications=("garch", "gjr"),
    window=400,          # rolling estimation window
    refit_every=1,       # 1 = exact protocol. Larger = faster approximation.
    n_lags=3,
)

print(features.names)     # which columns you got
print(features.note)      # says so explicitly if refit_every > 1

train_set, test_set = features.split(0.8)   # CHRONOLOGICAL, never random
```

### `refit_every` — speed versus exactness

`refit_every=1` re-estimates GARCH at every step. That is the exact protocol and it is slow (one maximum-likelihood fit per observation). `refit_every=k` re-estimates every k-th step and filters the recursion forward in between.

**The approximation never uses future data**, so it does not leak — but parameters lag by up to `k` observations. `features.note` says so in words whenever `k > 1`.

**This matters.** In this package's own GINN reproduction, the conclusion *flipped* between `refit_every=10` and `refit_every=1`. Use `1` for anything you will publish.

### Train the hybrid

```python
model = LSTMANNGARCH(
    features.features.shape[1],
    window=20,
    hidden_size=32,
    bidirectional=True,      # True gives the BLSTM-ANN-GARCH of Hu et al. 2020
)
history = model.fit_features(train_set, TrainConfig(lr=1e-3, max_epochs=200, patience=20))

print(model.evaluate(test_set))
# {'MAE': ..., 'RMSE': ..., 'R2': ...}
```

### Other hybrids

```python
from garchai.ensembles import GARCHXGBoost, StackedMLGARCH, MultiGARCHTransformer

# Residual correction: the booster fits what GARCH got WRONG, so it cannot
# take credit for what GARCH already knew.
xgb = GARCHXGBoost().fit(train_set)
print(xgb.evaluate(test_set))     # includes 'baseline_RMSE' — the gain is explicit

# Stacking. Reports whether it beat its own best base learner.
stack = StackedMLGARCH(n_folds=5).fit(train_set)
metrics = stack.evaluate(test_set)
print(metrics["beats_best_base"])   # if False, the ensemble added nothing
```

**`beats_best_base` exists because stacking often fails on short samples.** On the bundled S&P 500 example it returns `False`. The library reports that rather than quoting only its own error.

---

## Tutorial 5 — Comparing models properly

Comparing forecast errors without a significance test tells you nothing. Two models differing by 2% RMSE on 300 observations are indistinguishable.

```python
import numpy as np
import pandas as pd
from garchai.evaluation import diebold_mariano, model_confidence_set

# Per-observation losses on the SAME test set, one column per model.
losses = pd.DataFrame({
    "GARCH":    (actual - garch_pred) ** 2,
    "GARCHNN":  (actual - nn_pred) ** 2,
    "LSTM-ANN": (actual - hybrid_pred) ** 2,
})

# Pairwise: is GARCHNN significantly better than GARCH?
# Takes two loss ARRAYS, not the frame.
dm = diebold_mariano(losses["GARCHNN"].to_numpy(), losses["GARCH"].to_numpy(), horizon=1)
print(dm)     # Harvey-corrected, for the small samples these tests usually run on

# Which models survive as "best" at 90% confidence?
# NOTE: takes a (T, M) matrix or DataFrame -- one COLUMN per model -- not a dict.
mcs = model_confidence_set(losses, alpha=0.10, n_bootstrap=5000, block_length=5, seed=0)
print(mcs)    # DataFrame: model, rank, mean_loss, p-value, in_mcs
```

The MCS usually keeps **more than one** model. That is the point: it reports the set that cannot be distinguished from the best, rather than crowning a winner the data does not support.

**Use QLIKE, not MSE, for volatility.** MSE on variance is dominated by a handful of crisis days. QLIKE is the standard robust loss:

```python
from garchai.losses import qlike
loss = qlike(actual_variance, predicted_variance)
```

---

## Tutorial 6 — Publication tables and figures

```python
from garchai.reporting import (
    estimation_table, comparison_table, use_journal_style,
    plot_volatility, plot_var_exceedances, plot_training_history,
)
import matplotlib.pyplot as plt

use_journal_style()          # apply the package's Matplotlib style

# Estimation table. NOTE: this takes {name: FitResult} -- the object fit()
# RETURNS -- not the model objects themselves.
garch_result = model.fit(returns, train)
gjr_result = gjr_model.fit(returns, train)
table = estimation_table({"GARCH": garch_result, "GJR": gjr_result})
print(table.to_markdown())
open("table1.tex", "w").write(table.to_latex())     # booktabs LaTeX

# Comparison table — marks the best value per column
comp = comparison_table(
    {"GARCH": {"RMSE": 0.54, "MAE": 0.41}, "GARCHNN": {"RMSE": 0.49, "MAE": 0.38}},
    lower_is_better=("RMSE", "MAE"),     # be explicit — see warning below
)
print(comp.to_markdown())
```

**Warning about `comparison_table`.** It bolds the best value in each column, and `lower_is_better` decides which direction wins. Any column not listed is treated as higher-is-better. **Do not pass columns that have no "best"** — parameter values, p-values, or descriptive statistics. Bolding the maximum of a p-value column asserts a ranking that does not exist. Build those with `Table(frame=...)` directly.

Figures:

```python
# plot_volatility takes ONE volatility array and draws the +/- sigma band
fig = plot_volatility(returns, garch_result.conditional_volatility)
fig = plot_var_exceedances(returns, var, alpha=0.05)
fig = plot_training_history(history)
plt.savefig("figure1.png", dpi=300, bbox_inches="tight")
```

---

## 5. Complete model catalogue

**21 models built in full**, 3 delegated to `hybridecon`. Every entry names its paper and is runnable as written.

### 5.1 Classical econometric baselines

Validated against `arch` to 1e-12. Use these as the benchmark every neural model must beat.

```python
from garchai.econometric import GARCH, GJRGARCH, EGARCH

model = GARCH(omega=0.05, alpha=0.1, beta=0.85)
model = GJRGARCH()      # leverage effect
model = EGARCH()        # log-variance, no positivity constraint needed
```

### 5.2 Differentiable cells (build your own)

```python
from garchai.cells import GARCHCell, GJRCell, EGARCHCell, ComponentGARCHCell

cell = GARCHCell(omega=0.05, alpha=0.09, beta=0.88)
h, state = cell.filter(eps)                # autograd-tracked variance path
print(cell.persistence(), cell.unconditional_variance())

# Component GARCH: separates permanent and transitory volatility
cell = ComponentGARCHCell(rho=0.99, alpha=0.06, beta=0.80)
parts = cell.decompose(eps)                # {'permanent', 'transitory', 'variance'}
```

`ComponentGARCHCell` enforces `alpha + beta < rho` structurally, so the transitory component always decays faster than the permanent one.

### 5.3 Structural models — the network is *inside* the recursion

| Model | Paper | Import |
|---|---|---|
| `GARCHNN`, `StructuralGARCHLSTM` | Zhao et al. 2024 | `garchai.structural` |
| `RECH`, `SRNGARCH`, `SRNGJR`, `LSTMtGARCH` | Nguyen et al. | `garchai.structural` |
| `RealizedGARCH`, `DeepRGARCH` | Liu et al. 2023 | `garchai.structural` |
| `SigmaCell` + `-N`, `-RL`, `-NTV`, `-RLTV`, `SigmaLSTM` | Rodikov & Antulov-Fantulin | `garchai.structural` |
| `GARCHNet` | Buczynski & Chlebus 2023 | `garchai.structural` |
| `ANNGARCH` / `LikelihoodGARCHLSTM` | Liu & So 2020 | `garchai.structural` |
| `NeuralGARCH`, `NeuralBEKK` | Yin & Barucca | `garchai.structural` |
| `EmbeddedGARCHGRU`, `EmbeddedGARCHLSTM` | Wei et al. | `garchai.structural` |
| `AmortizedGARCHInference` | De Clerk & Savel'ev 2022 | `garchai.structural` |

All share the standard interface:

```python
from garchai.structural import GARCHNN, RECH, SigmaCell, GARCHNet, NeuralGARCH

model = GARCHNN(kernel="gjr")
model.fit(returns, TrainConfig(lr=1e-3, max_epochs=300))
model.summary(); model.forecast(horizon=5); model.var(0.05)
```

Two need extra arguments:

```python
from garchai.structural import DeepRGARCH, ANNGARCH

# Realized models take an intraday realized variance series
DeepRGARCH().fit(returns, train, realized_variance=rv)

# ANN-GARCH accepts exogenous predictors
ANNGARCH(n_exog=2).fit(returns, train, exog=X)
```

`AmortizedGARCHInference` is different — it learns a *parameter estimator*, replacing maximum likelihood:

```python
from garchai.structural import AmortizedGARCHInference

estimator = AmortizedGARCHInference(lag=15, n_samples=125_000, seed=0)
estimator.fit()                     # trains on the analytic moment map
params = estimator.estimate(returns)   # GARCHParameters, in milliseconds
```

### 5.4 Informed losses — `GINN`

The GARCH forecast enters the **loss**, not the architecture (Xu et al.).

```python
from garchai.informed import GINN, GINN0, build_ginn_inputs

data = build_ginn_inputs(returns, window=90, refit_garch_every=1)
model = GINN(lam=0.01, window=90, hidden=256, layers=3)
history = model.fit(data, TrainConfig(lr=1e-3, max_epochs=200), batch_size=128)

# lam=0 is the pure data-driven ablation:
ablation = GINN0(window=90)
```

`lam` weights the GARCH-consistency penalty. **Set `refit_garch_every=1` for published work** — see the warning in Tutorial 4.

### 5.5 Graph and decomposition — `GENSHIN`

VMD signal decomposition + a 60-model GARCH feature bank + a multi-scale graph network (Yu et al. 2025).

```python
from garchai.graph import GENSHIN, variational_mode_decomposition

modes, freqs = variational_mode_decomposition(signal, n_modes=4, alpha=2000)

model = GENSHIN(n_modes=4, n_components=6, sequence_length=20)
features = model.build_features(returns, target)   # fits up to 60 GARCH variants
print(features.n_models_fitted, features.narrowed, features.note)

model.fit(features, TrainConfig(lr=1e-3, max_epochs=100))
print(model.evaluate(features))
```

Narrowing the grid for speed is recorded in `features.narrowed` and `features.note` rather than hidden.

### 5.6 Feature hybrids and ensembles

```python
from garchai.ensembles import (
    build_garch_features, LSTMANNGARCH, GARCHXGBoost,
    StackedMLGARCH, MultiGARCHTransformer, GARCHTFT,
)
```

See [Tutorial 4](#tutorial-4--feature-hybrids-garch--machine-learning). `GARCHTFT` (Petrosino et al. 2025) is a full Temporal Fusion Transformer:

```python
from garchai.ensembles import GARCHTFT, historical_volatility, garman_klass

hv = historical_volatility(close, window=10)
gk = garman_klass(high, low, open_, close)      # needs intraday OHLC

model = GARCHTFT(window=20, d_model=32, n_heads=4, quantiles=(0.1, 0.5, 0.9))
model.names = ("return", "lagged_proxy", "garch")
model.fit(features, target, train=TrainConfig(lr=1e-3, max_epochs=250))

print(model.variable_importance(features))   # which inputs it ACTUALLY used
print(model.quantile_crossing_rate(features))# validity check — read this first
print(model.evaluate(features, target))

# use_garch=False gives the stand-alone TFT baseline
ablation = GARCHTFT(use_garch=False)
```

**Always read `quantile_crossing_rate` before quoting `interval_coverage`.** Nothing forces `q10 <= q90` — the pinball loss is separable across quantiles — and a crossed interval is empty, which makes any coverage figure from it meaningless.

### 5.7 Multivariate — `DCCGARCHMANN`

Both hybridisation directions from Fatima & Uddin 2022:

```python
from garchai.multivariate import DCCGARCHMANN, fit_dcc
import numpy as np

panel = np.column_stack([sp500_returns, ftse_returns, nikkei_returns])

# Standalone two-stage DCC (Engle 2002)
dcc = fit_dcc(panel)
print(dcc.a, dcc.b, dcc.persistence)
print(dcc.pairwise(0, 1))          # the correlation path between series 0 and 1

# The hybrids
model = DCCGARCHMANN(direction="dcc_to_mann").fit(panel)   # DCC feeds the network
model = DCCGARCHMANN(direction="mann_to_dcc").fit(panel)   # network feeds DCC
print(model.evaluate(panel))
```

### 5.8 Fuzzy — `GARCHFIS`

Li & Zhang 2026: a fixed Takagi-Sugeno rule base whose membership *widths* scale with the GARCH volatility forecast.

```python
from garchai.fuzzy import GARCHFIS

model = GARCHFIS(n_rules=6, kappa=1.0, window=250, refit_every=1)
model.fit(returns)
print(model.evaluate(returns))

print(model.membership_widths(sigma=0.5))   # the adaptation, made visible
print(model.membership_widths(sigma=2.0))   # wider in turbulent markets

ablation = GARCHFIS(kappa=0.0)              # adaptation OFF — the honest baseline
```

**On the bundled S&P 500 example, `kappa=1` does *not* beat `kappa=0`.** See [the reproduction log](https://github.com/merwanroudane/garchai/blob/main/docs/05_REPRODUCTION_LOG.md).

### 5.9 Reinforcement learning — `GARCHDDQNVaR`

Pokou et al. 2025: VaR as a classification problem solved by a Double DQN.

```python
from garchai.risk import GARCHDDQNVaR

agent = GARCHDDQNVaR(alpha=0.05, window=250, n_lags=5, refit_every=1, seed=0)
agent.fit(returns, n_episodes=25)

print(agent.rho)                      # class-imbalance scaling from Eq. 12
print(agent.predict_regime(returns))  # 0 = low risk, 1 = high risk
print(agent.var(returns))             # regime-conditional VaR
print(agent.evaluate(returns))        # classification + Kupiec + Christoffersen

# Learning curves for a paper figure
print(agent.history.reward, agent.history.recall, agent.history.epsilon)
```

The paper's prose and its Eq. 12 **disagree** on the false-negative penalty. The default follows the equation; `severe_false_negative=True` follows the prose. The contradiction is documented rather than silently resolved.

### 5.10 Sentiment

```python
from garchai.ensembles import (
    SentimentGARCHX, SentimentGARCHLSTM, normalise_sentiment, vix_sentiment_proxy,
)

sentiment = normalise_sentiment(raw_sentiment)     # maps to [0, 1]

model = SentimentGARCHX(sentiment_lags=1).fit(returns, sentiment)
print(model.params)                                 # includes gamma1
print(model.sentiment_contribution(sentiment))      # share of variance explained
```

**The sentiment series is a required argument — the library will not invent one.** The source paper's feed is proprietary. `vix_sentiment_proxy()` provides a documented public substitute, and any result from it is a proxy result, not a reproduction.

`sentiment_contribution` near zero means the augmentation is inert regardless of what the likelihood says.

### 5.11 Delegated to `hybridecon`

`FeatureGARCHLSTM` (Kim & Won 2018), `FeatureGARCHGRU`/`DLGARCH` (Michańków et al. 2023), `GARCHMIDASLSTM` (Ersin & Bildirici 2023) are plain feature hybrids already covered by `hybridecon`. `garchai` deliberately does not duplicate them.

---

## 6. Recipes

### Use your own CSV

```python
import pandas as pd, numpy as np

df = pd.read_csv("my_prices.csv", parse_dates=["date"], index_col="date")
returns = 100 * np.log(df["close"]).diff().dropna()     # PERCENT — see Concept 2
model = GARCH().fit(returns)
```

### Download fresh data

```python
from garchai.datasets import fetch_yahoo, log_returns

prices = fetch_yahoo("AAPL", start="2020-01-01", end="2024-12-31")   # needs [data]
returns = log_returns(prices["close"], percent=True)
```

### Reproducible runs

```python
from garchai.core import set_seed
set_seed(0)                        # numpy, torch, and Python random
```

### Rolling out-of-sample forecasting

```python
import numpy as np

predictions = []
for t in range(1000, len(returns)):
    m = GARCH()
    m.fit(returns[:t], TrainConfig(max_epochs=300))
    predictions.append(m.forecast(horizon=1).variance[0])
predictions = np.array(predictions)
```

This is slow and correct. It is what "out-of-sample" means.

### Check what a model claims about itself

```python
print(model.provenance)                      # human-readable
print(model.provenance.deviations)           # every departure from the paper
print(model.provenance.data_substitutions)   # every dataset swap
```

### List every paper and its build status

```bash
python -m garchai.papers.status
```

---

## 7. Running the tests

```bash
pip install "garchai[dev]"
git clone https://github.com/merwanroudane/garchai
cd garchai
```

| Command | Tests | Time |
|---|---|---|
| `pytest -m "not slow"` | 539 | **1m46s** |
| `pytest` | 551 | ~5m |
| `pytest --doctest-modules src/garchai` | 597 | **5m43s** |

```bash
# Fast loop while developing
pytest -m "not slow" -q

# One family
pytest tests/unit/test_frontier.py -v

# Numerical parity against `arch`
pytest tests/numerical/ -v
```

The `slow` tests are the parameter-recovery and training-convergence checks. They are the ones that would catch a real regression, so run the full suite before publishing anything.

---

## 8. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `mat1 and mat2 must have the same dtype` | float32 default with float64 data | `torch.set_default_dtype(torch.float64)` **before** building models |
| `Status: maximum epochs reached` | fit did not converge | `TrainConfig(max_epochs=1500, refine_with_lbfgs=True)` |
| Nonsense parameters, huge omega | returns in decimals not percent | multiply by 100, or `ModelConfig(scale="auto")` |
| `ModuleNotFoundError: arch` | optional dependency | `pip install "garchai[benchmarks]"` |
| Suspiciously high R² | look-ahead in your features | use `build_garch_features`, and split chronologically |
| `sentiment contains NaN or inf` | rolling proxy's first window is empty | seed the first value; do not pass `NaN` |
| `Both risk classes must be present` | `quantile_threshold` too extreme | raise it so some days are labelled high-risk |
| `alpha + beta < rho` error | component GARCH ordering violated | lower `alpha`/`beta` or raise `rho` |
| Fit is very slow | `refit_every=1` does one MLE per step | set `refit_every=25` for exploration, `1` for publication |

---

## 9. Provenance and honesty

This library is built for people who will **cite it in a paper**. Three commitments:

**1. Every model declares its lineage.** `model.provenance` gives the paper, the equations implemented, the code status, and every deviation. A derived status tool cross-references the registry against the source so it cannot silently drift.

**2. No code was ported from unlicensed repositories.** Seven of the nine reference repositories carry no licence file. Those models were implemented from their papers.

**3. Negative results are reported.** The [reproduction log](https://github.com/merwanroudane/garchai/blob/main/docs/05_REPRODUCTION_LOG.md) records what did *not* work, including:

- **`GARCHFIS` does not beat its own no-adaptation ablation** on S&P 500 returns — the paper's sole contribution does not replicate here.
- **`StackedMLGARCH` does not beat its best base learner** on the bundled example, and returns `beats_best_base: False` rather than quoting only its own error.
- **The sentiment model is not reproduced** — its feed is proprietary, and no substitute is presented as equivalent.
- **A GINN reproduction flipped its conclusion** between `refit_every=10` and `refit_every=1`, which is why that parameter is documented so prominently.

Further documentation:

- [Inventory and licence audit](https://github.com/merwanroudane/garchai/blob/main/docs/00_INVENTORY.md)
- [Coverage matrix — all 24 papers](https://github.com/merwanroudane/garchai/blob/main/docs/01_COVERAGE_MATRIX.md)
- [Architecture](https://github.com/merwanroudane/garchai/blob/main/docs/02_ARCHITECTURE.md)
- [Decisions and open gaps](https://github.com/merwanroudane/garchai/blob/main/docs/03_DECISIONS_AND_GAPS.md)
- [Reproduction log](https://github.com/merwanroudane/garchai/blob/main/docs/05_REPRODUCTION_LOG.md)

Runnable end-to-end examples live in [`examples/`](https://github.com/merwanroudane/garchai/tree/main/examples).

---

## 10. Citation

```bibtex
@software{roudane2026garchai,
  author  = {Roudane, Merwan},
  title   = {garchai: Econometrically Constrained, Differentiable,
             AI-Enhanced Conditional Heteroskedasticity Models},
  year    = {2026},
  url     = {https://github.com/merwanroudane/garchai},
  license = {MIT}
}
```

**Please also cite the original paper of any model you use.** Each one is listed in `model.provenance` and in the [coverage matrix](https://github.com/merwanroudane/garchai/blob/main/docs/01_COVERAGE_MATRIX.md).

---

## Licence

MIT © 2026 Merwan Roudane

Issues and contributions: [github.com/merwanroudane/garchai](https://github.com/merwanroudane/garchai)
