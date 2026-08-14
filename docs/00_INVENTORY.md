# GARCH-AI Folder Inventory — Phase 0

**Date of inspection:** 2026-08-14
**Inspected by:** full read of every file in the GARCH AI folder, per
`CLAUDE_GARCHAI_BUILD_REQUIREMENTS.txt`.

---

## 1. Specification / notes files

| File | Lines | Status |
|---|---|---|
| `garch_ai_ecosystem_complete_master.txt` | 3 024 | **Read in full.** 70 sections. Primary blueprint. |
| `garch_ai_ecosystem_complete_master (1).txt` | 3 024 | Byte-identical duplicate (`md5 6b0d4365…`). Safe to delete. |
| `garch_ai_ecosystem_complete_master.html` | — | HTML rendering of the same content. |
| `garch_lstm_best_papers_and_code_notes.txt` | 987 | **Read in full.** The 4-way GARCH-LSTM taxonomy. |
| `neural_garch_library_research_and_build_notes.txt` | 4 393 | **Read.** 141 sections; largely a superset/earlier draft of the master file. Unique content in §1–5 (Tier-S literature detail), §134 (paper→model map), §135 (gap analysis). |
| `CLAUDE_GARCHAI_BUILD_REQUIREMENTS.txt` | 74 | Governing requirements. |

---

## 2. Research papers — 21 PDFs, all extracted to text and identified

Every PDF was converted to text (PyMuPDF) and identified. **Deep-read** = full
methodology + equations + experimental protocol extracted. **Identified** =
title/authors/venue/scope confirmed; deep read scheduled at implementation time
per the workflow in master blueprint §70.

### Tier S — structural (deep-read)

| PDF | Paper | Read |
|---|---|---|
| `2402.06642v1.pdf` | Zhao, Zhu, Ng & Lee — *From GARCH to Neural Network for Volatility Forecast*, AAAI 2024 | **Deep** — Eqs 1–7 |
| `2504.09380v2.pdf` | Wei, Yang & Cui — *Unified GARCH-Recurrent Neural Networks*, 2025 (v2, 21 Nov 2025) | **Deep** — Eqs 14–29 |
| `2202.11285v1.pdf` | Yin & Barucca — *Neural Generalised ARCH*, 2022 | **Deep** — Eqs 1–22 |
| `2010.13061v2.pdf` | Nguyen, Tran & Kohn — *Recurrent Conditional Heteroskedasticity* (RECH), JAE 37(5) 2022 | **Deep** — Eqs 7–8, Thm 1 |
| `2302.08002v2.pdf` | Liu, Wang, Tran & Kohn — *Deep Learning Enhanced Realized GARCH* (DeepRGARCH / RealRECH) | **Deep** — Eqs 5–13 |
| `2205.07022v1.pdf` | Rodikov & Antulov-Fantulin — *Volatility-inspired σ-LSTM cell*, 2022 | **Deep** — Eqs 1–12 |
| `2309.01565v1.pdf` | Rodikov & Antulov-Fantulin — *Introducing the σ-Cell*, 2023 (66 pp) | Identified |
| `2410.00288v1.pdf` | Xu, Liechty, Benthall, Skar-Gislinge & McComb — *GARCH-Informed Neural Networks* (GINN) | **Deep** — Eqs 9–18 |
| `s10614-023-10390-7.pdf` | Buczyński & Chlebus — *GARCHNet: VaR Forecasting with GARCH Models Based on NNs*, Comput. Econ. 2023 | Identified |

### Tier A/B — feature hybrids, benchmarks, extensions

| PDF | Paper |
|---|---|
| `kim2018.pdf` | Kim & Won (2018), *Expert Syst. Appl.* 103:25–37 — LSTM + multiple GARCH-type models |
| `hu2020.pdf` | Hu, Ni & Wen (2020), *Physica A* 557:124907 — LSTM-ANN + GARCH, copper price |
| `2310.01063v1.pdf` | Michańków, Kwiatkowski & Waszkowski (2023) — Combining Deep Learning and GARCH for volatility & risk |
| `mathematics-11-01785-v2.pdf` | Ersin & Bildirici (2023), *Mathematics* 11:1785 — GARCH-MIDAS-LSTM |
| `s00521-025-11468-z.pdf` | Petrosino, Bacco, Salvati, Merone & Papi (2025), *Neural Comput. Applic.* — GARCH-Temporal Fusion Transformer |
| `s00521-022-07631-5.pdf` | Fatima & Uddin (2022), *Neural Comput. Applic.* — DCC-GARCH + multivariate ANNs |
| `ijfs-13-00155.pdf` | Maingo, Ravele & Sigauke (2025), *IJFS* 13:155 — fusion of statistical and ML models |
| `s10614-025-11042-8.pdf` | (2026) *Comput. Econ.* 67:313–353 — sentiment-based hybrid GARCH-LSTM, volatility + VaR |
| `s10614-025-11191-w.pdf` | (2025) *Comput. Econ.* — hybrid VaR: GARCH + Double Deep Q-Network |
| `2603.14793v1.pdf` | Li & Zhang (2026) — GARCH-FIS, fuzzy inference with volatility-driven parameter adaptation |
| `mathematics-14-00624.pdf` | Rubio, Alba, Velasquez & Ramos (2026), *Mathematics* 14:624 — **Stacked ML-GARCH for Bitcoin VaR** |
| `9834-21576-1-PB.pdf` | Gebreslase, Mutua, Fissuh & Girmay (2026), *Commun. Math. Biol. Neurosci.* 2026:48 — **Hybrid Multi-GARCH Transformer** |

> The last two are **not referenced anywhere in the ecosystem master file**. They
> are new material and have been added to the registry.

---

## 3. Source-code archives — 9 zips, all extracted

All nested archives were also unpacked (`RECH-main` contained a 25 MB `.7z` plus
three inner `.zip` files that had not been opened).

| Repo | Provenance | Language | License | Contents after full extraction |
|---|---|---|---|---|
| `RECH-main` | **OFFICIAL** (VBayesLab/RECH) | MATLAB | **none** | `SP500_analysis.m`, `SMC_for_GARCH/`, `SMC_for_LSTM_tGARCH/` (SMC + likelihood/data annealing, one-step forecast, residual analysis), plus *"Does anything beat LSTM-tGARCH?"* companion code for large-cap, mid-cap and index panels. `.mat` results + `individual_stock_data.mat`, `realized_data.mat`. |
| `RealRECH-main` | **OFFICIAL** (VBayesLab/RealRECH) | MATLAB | **none** | ⚠️ **Contains only `SMC_for_RECH/`** (RNN-tGARCH). **No DeepRGARCH implementation.** README: *"The Python version will be uploaded soon."* |
| `tsgarch-main` | R package source (tsmodels) | R / C++ / TMB | (CRAN, DESCRIPTION) | Full modern GARCH framework — reference for filters, likelihood, initialization, persistence, news impact, backtest. |
| `garchnet-replication-main` | COMMUNITY_REPLICATION | Python | **none** | `garchnet.py`, `garch_baseline.py`, `skewed_t.py`, `data_loader.py`, `analyze.py`, `plots.py`, `format_table.py`, tests, `STATUS.md`. Best-organised community repo in the set. |
| `GLoVE-main` | COMMUNITY (GINN-inspired) | Python | **none** | Alpha101 features, LSTM / TFT / TSMixer trainers, GARCH-informed loss experiments, `reproduction.md`. |
| `garch-tft-main` | PAPER_LINKED (lbacco) | Python | **none** | ⚠️ Partial. README says code will be released post-acceptance. Present: `GARCH/fittinggarch.py`, `MODELS/anngarch.py`, `MODELS/lstm_garch.py`, `TFTS/script_*`. |
| `GARCH-LSTM-main` | INDEPENDENT_RELATED | Python | MIT | GARCH/EGARCH/GJR/APARCH → LSTM on USD/BRL realized volatility (21-day rolling σ). |
| `LSTM_GARCH-1-master` | COMMUNITY_REPLICATION of Kim & Won | Python (notebook) | MIT | `model.ipynb`, PyFlux + Keras. |
| `neural-garch-main` | INDEPENDENT_RELATED | Python | **none** | ⚠️ **Not** Yin & Barucca. Implements a generic `σ²_t = f_θ(r²_{t−1}, σ²_{t−1})` net; docs in Chinese. Low reference value for `NeuralGARCH`. |

### License consequence

Seven of nine repositories carry **no license file**. Per master blueprint §63,
this means: *do not port their code*. Every model will be implemented
independently from the paper equations; repository code is used only for
**behavioural comparison, hyper-parameter confirmation and numerical
validation**. The two MIT repos (`GARCH-LSTM-main`, `LSTM_GARCH-1-master`) are
the only ones from which code could legally be adapted, and neither is needed.

---

## 4. Local environment (verified)

Python 3.11.0 — `numpy` 1.26.4, `scipy` 1.13.1, `pandas` 2.3.3, **`torch`
2.10.0+cpu**, **`arch` 8.0.0**, `statsmodels` 0.14.6, `scikit-learn` 1.8.0,
`matplotlib` 3.10.8, `seaborn` 0.13.2, `xgboost` 3.2.0, **`yfinance` 1.1.0**,
`pytest` 9.0.3. Network access confirmed (PyPI reachable).

No GPU. All Tier-S models are small (scalar-to-64-dim recurrent states); CPU
training is adequate, but SMC-based Bayesian inference (RECH, RealRECH) will be
the slow path and needs a vectorised-particle implementation.
