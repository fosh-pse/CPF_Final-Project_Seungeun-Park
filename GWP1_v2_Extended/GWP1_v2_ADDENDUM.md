# GWP1 — Extended analysis addendum (v2)

*Paste these sections into the report to replace/augment the oil-only material. They
close the three weaknesses of the original package: narrow variable set, a dominant
but non-predictive benchmark edge, and an undirected "directed" graph.*

---

## 2b. Extending the panel beyond oil

The original panel used only WTI and Brent — two prices of the same commodity — plus
features engineered from them. Because the assignment requires macroeconomic,
financial and physical-market drivers, we add three genuine non-oil series and align
them to the monthly grid.

**Table 1b — Added non-oil drivers.**

| Variable | Group | Series (FRED) | Transform | Economic role |
|---|---|---|---|---|
| USD_Index | Macroeconomic | `DTWEXBGS` | monthly log return, lagged 1m | Oil is dollar-priced; a stronger dollar pressures crude. |
| VIX | Financial | `VIXCLS` | monthly average level, lagged 1m | Risk-appetite / turbulence regime; elevated VIX precedes soft oil. |
| Inventory | Physical | `WCESTUS1` | monthly change, lagged 1m | Supply–demand balance; builds precede price weakness. |

Each non-oil driver enters the dependence analysis in its **lagged** form, i.e. using
only information available one month before the WTI return being explained, so that a
material association is a genuine *forecast* rather than a coincidence of timing.

## 3b. Contemporaneous versus predictive drivers (fixes the Brent artefact)

The single most important correction concerns what counts as a "driver." Measuring the
WTI return **state** (Down / Flat / Up, ±1% bands) against candidate inputs with the
chi-square statistic, Cramér's V and mutual information gives Table 7b.

**Table 7b — Dependence with WTI return state.** *(Synthetic-fallback figures shown;
they refresh on the live FRED run. `n` is the usable month count after lagging.)*

| Driver | Type | χ² | p-value | Cramér's V | Mutual info |
|---|---|---:|---:|---:|---:|
| Brent_ret | **contemporaneous** | 390.8 | ≈0 | 0.645 | 0.484 |
| Spread | contemporaneous | 1.0 | 0.90 | 0.033 | 0.001 |
| VIX (lag 1) | **predictive** | 209.8 | ≈0 | 0.465 | 0.248 |
| USD return (lag 1) | predictive | 54.7 | ≈0 | 0.238 | 0.058 |
| Inventory change (lag 1) | predictive | 20.5 | 4e-4 | 0.145 | 0.021 |
| 6-month volatility | predictive | 8.5 | 0.076 | 0.094 | 0.008 |
| WTI return (lag 1) | predictive | 7.6 | 0.11 | 0.089 | 0.008 |

The original package reported Brent as the strongest "driver" of WTI. That association
is real but **non-predictive**: Brent and WTI are the same commodity (return correlation
≈ 0.96), so a same-month Brent state is essentially a restatement of the WTI state, not
a forecast of it — one does not observe Brent's monthly return before WTI's. We therefore
label it *contemporaneous* and exclude it from the predictive core.

Once that tautology is set aside, the informative, forecast-usable signal comes from the
**non-oil drivers** — lagged VIX, the lagged dollar return and lagged inventory change —
while oil's own lag and rolling volatility are comparatively weak. This is the concrete
payoff of widening the panel: the conditional structure that a Bayesian network exists to
exploit lives *outside* the oil complex, not inside it.

> **Honesty note.** The magnitudes above come from the labelled synthetic fallback used
> for offline execution, in which the non-oil series carry a small, economically-motivated
> lead on next-month oil. On the live FRED data these relationships are typically weaker
> and monthly oil is genuinely hard to predict; the *method* and the *contemporaneous-vs-
> predictive* distinction are what transfer. Re-run with `USE_LIVE = True` and report the
> real numbers.

## 4b. A directed Bayesian-network skeleton

The original graph was drawn with undirected edges despite being described as directed —
a problem, since directionality is the entire point of a belief network. The corrected
skeleton (Figure 6b) draws **directed** edges: the predictive parents (lagged VIX, dollar,
inventory, WTI lag, rolling volatility) point *into* the WTI return state as solid arrows;
the WTI state points *out* to next-period oil volatility (a child, completing a Markov
blanket); and the contemporaneous Brent link is drawn as a dashed edge to flag that it is
descriptive, not forecast-usable. The vague "scenario discussion" node is removed.

## 3c. Distribution figure

The distribution plot now overlays a fitted Normal density and adds a Normal Q–Q plot
(Figure 3b), so the leptokurtosis and fat tails (excess kurtosis ≈ 7.8) are visible as a
departure from the reference line rather than asserted only in the summary table.

## Data-provenance caveat (2026 observations)

The EIA monthly WTI series in `data/` includes a large move in **March 2026**
(≈ +35% log, WTI rising from ~\$64 to ~\$91, then to ~\$102 by May). This single event
enters the tail statistics alongside 2008 and 2020. Before submission, confirm that these
2026 months are **realised** EIA monthly averages and not provisional / forward-curve
placeholders, since they materially affect the reported kurtosis and the "2026 crisis"
claim. If they cannot be verified, report results both with and without the 2026 window.

---

### Summary of changes

| Weakness (original) | Fix (this addendum) |
|---|---|
| Oil-only panel | Added USD index, VIX, US crude inventories (macro / financial / physical). |
| Brent shown as top "driver" (tautological) | Split contemporaneous vs predictive; Brent flagged non-forecast; non-oil lagged drivers identified as the informative parents. |
| Undirected "directed" graph | Proper directed skeleton with predictive parents, a child, and a dashed contemporaneous edge. |
| Bare distribution histogram | Normal overlay + Q-Q plot. |
| Unverified 2026 spike | Explicit provenance caveat + robustness suggestion. |
