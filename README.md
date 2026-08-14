# garchai

**Econometrically constrained, differentiable, AI-enhanced conditional heteroskedasticity models.**

[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![PyTorch](https://img.shields.io/badge/backend-PyTorch-ee4c2c)](https://pytorch.org)

`garchai` is a research-grade framework for volatility models in which
econometric structure is a **first-class differentiable component of the neural
model itself** — not a preprocessing step that produces a feature.

The distinction is the whole point:

```
NOT:  GARCH  →  precomputed σₜ  →  arbitrary neural network
BUT:  econometric recurrence  ↔  neural computation,
      trained jointly, transparently, under valid constraints
```

---

## Why another volatility package?

Because "GARCH-LSTM" names at least four genuinely different algorithms, and
most software collapses them into one class. `garchai` keeps them apart:

| Family | What the network actually does | Reference |
|---|---|---|
| `FeatureGARCHLSTM` | GARCH fitted first; σₜ, zₜ become inputs | Kim & Won (2018) |
| `StructuralGARCHLSTM` | GARCH translated *into* the architecture | Zhao et al., AAAI 2024 |
| `EmbeddedGARCHLSTM` | GARCH recursion inside the recurrent gates | Wei, Yang & Cui (2025) |
| `GARCHInformedLSTM` | GARCH enters the *loss*, not the forward pass | Xu et al. (2024) |

A package that exports a single `GARCHLSTM` cannot tell you which one you got.

**Scope note.** `garchai` deliberately does not duplicate
[`hybridecon`](https://github.com/merwanroudane/hybridecon), which already owns
the feature-hybrid family. Those models are reached here through an adapter.
`garchai` owns the structural, embedded, informed, Bayesian and
constrained-multivariate families.

---

## Status

**Tier S complete.** All five structural papers, plus GINN and the six-model
sigma-Cell family. 8 of the 24 papers in the collection are built, 3 are
delegated to `hybridecon`, 13 remain. See [`docs/02_ARCHITECTURE.md`](docs/02_ARCHITECTURE.md) for the full
build plan.

Shipping today:

- **Differentiable cells** — `GARCHCell`, `GJRCell`, `EGARCHCell`, validated
  against [`arch`](https://github.com/bashtage/arch) to **4.4 × 10⁻¹⁵** on the
  full conditional-variance path
- **Constraint engine** — softmax-simplex transforms that guarantee
  `ω > 0, α, β ≥ 0, α + β < 1` for *any* real parameter vector, at every time step
- **Distributions** — Normal, standardised Student-t, Hansen skewed-t, GED, each
  verified to integrate to 1 with zero mean and unit variance
- **Models** — `GARCH`, `GJRGARCH`, `EGARCH` by joint maximum likelihood
- **Risk** — VaR, ES, Kupiec, Christoffersen, dynamic quantile, McNeil–Frey
- **Evaluation** — Diebold–Mariano, Giacomini–White, Model Confidence Set
- **Reporting** — journal-style LaTeX/Markdown tables and figures
- **Structural models** — Zhao et al. (AAAI 2024) `GARCHNN` and
  `StructuralGARCHLSTM` with pluggable GARCH / GJR / FI-GARCH kernels; Wei et al.
  (2025) `EmbeddedGARCHGRU` and `EmbeddedGARCHLSTM`
- **Variational neural GARCH** — Yin & Barucca (2022) `NeuralGARCH` with
  time-varying `(ωₜ, αₜ, βₜ, νₜ)`, and `NeuralBEKK`, whose covariance is
  symmetric positive definite **by construction** at every step
- **Recurrent conditional heteroskedasticity** — Nguyen et al. (2022) `RECH`,
  `SRNGARCH`, `SRNGJR`, `LSTMtGARCH`; Liu et al. (2023) `DeepRGARCH` /
  `RealRECH` with the `RealizedGARCH` baseline it extends
- **σ-Cell family** — Rodikov & Antulov-Fantulin's `SigmaLSTM` (2022) and all
  five 2023 variants (`SigmaCell`, `-N`, `-RL`, `-NTV`, `-RLTV`)
- **GARCH-informed learning** — Xu et al. (2024) `GINN` and `GINN0`
- **Real data** — nine daily series (1990–2026) bundled, no network needed

Run `python -m garchai.papers.status` for the live build status — it is derived
from the code, not a hand-kept list, so it cannot drift.

---

## Install

```bash
pip install -e ".[all]"
```

---

## Sixty-second tour

```python
from garchai import GARCH
from garchai.core import ModelConfig
from garchai.datasets import load_returns

returns = load_returns("sp500", start="2010-01-01", end="2019-12-31")

model = GARCH(config=ModelConfig(mean="constant", distribution="student-t"))
result = model.fit(returns.to_numpy())

print(model.summary())
print(model.forecast(horizon=10).to_frame())
print(model.var(alpha=0.01)[-5:])          # Value-at-Risk, as a return level
```

Then run the full worked example, which fits four models, compares them
formally, backtests the risk numbers and writes four figures and three tables:

```bash
python examples/01_getting_started.py
```

---

## Design commitments

These are the rules the code is actually held to, not aspirations.

**Nothing is silently substituted.** `arch` and `tsgarch` use genuinely different
variance-initialisation conventions, so `garchai` implements *both* — named
`arch_backcast` and `tsgarch_backcast` — rather than splitting the difference.
EGARCH has no closed-form multi-step forecast, so `forecast(method="analytic")`
**raises** rather than quietly simulating.

**Targets are named after their papers.** Zhao et al. and Wei et al. both say
"5-day realized volatility" and mean different estimators — one is undivided,
undemeaned and includes period *t*; the other is divided, demeaned and strictly
lagged. They differ by a factor of about 2.2. Both ship, separately, as
`realized_volatility_zhao` and `realized_volatility_wei`.

**Gradients are never truncated by accident.** Cell state is detached only when
`truncate_bptt` is passed explicitly, and there is a test asserting the two give
different gradients.

**Every model carries its provenance.**

```python
>>> GARCH.provenance.implementation_status
<ImplementationStatus.PAPER_FAITHFUL: 'PAPER_FAITHFUL'>
```

Paper, equations, official repository, implementation status and every
documented deviation are attributes of the class, printed by `.summary()`.

**Real data only in examples.** Simulated series appear in exactly one place —
training the amortized estimator of De Clerk & Savel'ev (2022), where the
training set *is* an analytic moment map.

**No code is ported from unlicensed repositories.** Seven of the eleven
reference repositories carry no licence, so every model is written from the
published equations; repository code is used for behavioural comparison and
numerical validation only.

---

## Documentation

| Document | Contents |
|---|---|
| [`docs/00_INVENTORY.md`](docs/00_INVENTORY.md) | Every paper and repository inspected, with licence status |
| [`docs/01_COVERAGE_MATRIX.md`](docs/01_COVERAGE_MATRIX.md) | Paper ↔ model ↔ code ↔ build decision, all 26 models |
| [`docs/02_ARCHITECTURE.md`](docs/02_ARCHITECTURE.md) | Package structure, the two primitives, the 8-phase plan |
| [`docs/03_DECISIONS_AND_GAPS.md`](docs/03_DECISIONS_AND_GAPS.md) | Decisions taken, data substitutions, open gaps |
| [`docs/05_REPRODUCTION_LOG.md`](docs/05_REPRODUCTION_LOG.md) | Empirical reproduction attempts, including the one that did **not** reproduce |
| `src/garchai/resources/papers.yaml` | Machine-readable registry of 24 papers |
| `src/garchai/resources/repos.yaml` | Registry of reference code, with licence audit |

---

## Roadmap

| Phase | Contents | Status |
|---|---|---|
| 1 | Econometric core: cells, constraints, distributions, risk, evaluation | ✅ done |
| 2 | `GARCHNN`, `StructuralGARCHLSTM` (Zhao); `EmbeddedGARCHGRU/LSTM` (Wei) | ✅ done |
| 3 | `NeuralGARCH` (Yin) ✅; `RECH` (Nguyen) ✅ | ✅ done |
| 4 | `SigmaLSTM` ✅, `SigmaCell` ✅, `GINN` ✅, `GARCHNet` | in progress |
| 5 | `RealRECH` / `DeepRGARCH`, realized-measure infrastructure | ✅ done |
| 6 | `NeuralBEKK`, `NeuralDCC`, `DCCGARCHMANN` | |
| 7 | Feature hybrids, ensembles, `GARCHTFT`, `MultiGARCHTransformer` | |
| 8 | `GARCHFIS`, `GARCHDDQNVaR`, regimes, panel, `GENSHIN` | |

---

## Citation

```bibtex
@software{roudane_garchai,
  author = {Roudane, Merwan},
  title  = {garchai: Econometrically Constrained, Differentiable,
            AI-Enhanced Conditional Heteroskedasticity Models},
  url    = {https://github.com/merwanroudane/garchai},
  year   = {2026}
}
```

Please also cite the paper behind whichever model you use — every class names it
in `.provenance`.

---

## Licence

MIT. See [LICENSE](LICENSE).
