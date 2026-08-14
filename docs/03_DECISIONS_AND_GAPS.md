# Decisions taken, and remaining gaps

Supersedes the earlier "missing resources" list. Everything below is **decided**;
nothing here is waiting on an answer.

---

## 1. Blockers cleared

You supplied the three blocking papers on 2026-08-14. All are read and
registered in `papers.yaml`.

| Was blocking | Paper now held | Unblocks |
|---|---|---|
| A3 | Liu & So (2020), *Information* 11(10):489 | `ANNGARCH`, `LikelihoodGARCHLSTM` |
| A4 | Yu, Xiao, Zhu & Zhang (2025), *Finance Research Letters* 78:107196 | `GENSHIN`, the whole `graph/` module |
| A5 | De Clerk & Savel'ev (2022), *Physica A* 603:127869 | `NeuralGARCHEstimator`, `AmortizedGARCHInference` |

**No paper is now blocking any model on the build plan.**

---

## 2. Standing data rule

> **When a paper's dataset is unavailable, substitute the closest equivalent
> public dataset, implement the model faithfully on the substitute, and record
> the substitution explicitly.**

Faithfulness applies to the *model*, not to the *data vendor*. Every
substitution is declared in three places: the dataset loader docstring, the
model's `deviations` metadata, and a "Data substitution" note under the
reproduction table so no reader can mistake it for the original sample.

Applied cases:

| Paper | Original data | Substitute | Why |
|---|---|---|---|
| RealRECH / DeepRGARCH | Oxford-Man Realized Library, 31 indices, 5-min RV, 2004–2022 | Realized measures rebuilt from public intraday data for the overlapping indices (SPX, DJI, GDAXI, AEX where available); same 5-min sampling, same Hansen–Lunde scaling `ĉ = Σy²/Σrv` | Oxford-Man was discontinued in 2022 and is no longer served |
| GENSHIN | 8 Chinese indices, 5-min RV, M=48 | Same RV estimator applied to indices with obtainable 5-min data; Chinese indices retained where sourceable, otherwise matched by market-cap tier and liquidity | Chinese exchange intraday data is not on public feeds |
| Liu & So 2020 | 17 Bloomberg Terminal fundamentals for the S&P 500 | Public analogue feature set: FRED macro series (term spread, credit spread, VIX, industrial production) plus index-level valuation proxies, same 20-day lookback and same count where possible | Bloomberg fundamentals are licensed and cannot be redistributed |
| Sentiment GARCH-LSTM | proprietary media/social sentiment | Public sentiment proxy (news-volume / open sentiment index), clearly labelled | vendor-licensed |

Where a substitution makes a published number unreproducible, the reproduction
test asserts *qualitative* agreement (model ranking, sign, order of magnitude)
rather than the exact figure, and says so in the test docstring.

---

## 3. Project decisions taken

| # | Decision | Rationale |
|---|---|---|
| 1 | **Bundle real data.** Compressed daily CSVs ship in `src/garchai/datasets/data/` for S&P 500, DJI, NASDAQ, EUR/USD, gold and Bitcoin — the exact series the Tier-S papers use — with a `fetch_yahoo()` helper for fresh data. | Requirements forbid simulated teaching examples; live fetching makes docs fragile and non-reproducible. |
| 2 | **Papers and archives are git-ignored.** They stay locally as the research corpus; the repo ships `papers.yaml` with DOIs and links instead. | ~130 MB, and most are paywalled publisher PDFs that must not be redistributed. |
| 3 | **MIT license.** | Standard for scientific Python; imposes nothing on future work. |
| 4 | **No R companion package.** | Out of scope for v0.1. `rugarch`/`rmgarch`/`tsgarch` remain read-only references. |
| 5 | **No code is ported from any reference repository.** Every model is written from the published equations. | 7 of 9 repos have no license file (blueprint §63). Repo code is used for behavioural comparison, hyper-parameter confirmation and numerical validation only. |

---

## 4. Remaining gaps — worked around, not blocking

**RealRECH official code.** `RealRECH-main.zip` ships only `SMC_for_RECH/`
(RNN-tGARCH); the DeepRGARCH implementation is absent and the repo README still
says the Python version is forthcoming. The paper (Eqs 7a–7f, likelihood 8) is
complete enough to implement faithfully. What it does not pin down numerically is
the SMC configuration — tempering schedule, ESS threshold, particle count,
Markov-kernel choice. Resolution: adopt the settings from the RECH SMC code we
*do* hold (same authors, same framework), and record them as `deviations`.

**Community repos not downloaded.** `yeasinstat/GarchInformedLSTM`,
`jhihan/GARCH_Neural_Network`, `czb9829/GARCH-MIDAS`, `aryanbhambu/GaMM`, and
the DCC/XGBoost repos listed in `repos.yaml`. None is on the critical path and
all are unlicensed, so their only permitted use is behavioural comparison. I'll
clone any of them if a specific model turns out to need a hyper-parameter check.

**csGARCH is not in `arch`.** GENSHIN's 60-model feature bank needs the
component-sGARCH of Engle & Lee (1999). Being built in `garchai.econometric`.

**Papers listed in the blueprint but never supplied** — *Array* 2026 stacking
paper, *Risks* 10(12):237, Kakade et al. 2022. All benchmark-breadth only; the
ensemble slot is covered by Rubio et al. (2026), which you did supply.
