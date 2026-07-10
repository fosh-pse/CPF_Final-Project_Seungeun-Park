# Report snippet — paste into Section 6 (Dependence Screening)

*Add this after the existing contemporaneous dependence table. It operationalises
the "contemporaneous vs predictive" distinction the report already discusses, using
the real 2006–2025 panel.*

---

**Contemporaneous versus one-month-ahead dependence.** The dependence screen above
measures each candidate driver in the *same* month as the WTI return. That is the
right test for confirming co-movement, but it does not establish whether a driver
can be used to *forecast* the next month, because a same-month value is not observed
before the return it is being compared with. To separate the two, we re-run the
screen on the one-month-lagged (lag-1) version of each non-oil driver.

**Table 7b — Same-month co-movement vs one-month-ahead prediction (real panel,
2006–2025, 239 months).**

| Driver | Timing | χ² | p-value | Cramér's V | Sig. 5% |
|---|---|---:|---:|---:|:--:|
| Dollar index return | contemporaneous | 20.23 | 0.0005 | 0.206 | ✔ |
| Dollar index return | lag 1 (predictive) | 2.49 | 0.65 | 0.072 | ✗ |
| VIX change | contemporaneous | 4.79 | 0.31 | 0.100 | ✗ |
| **VIX change** | **lag 1 (predictive)** | **10.76** | **0.029** | **0.150** | **✔** |
| Crude inventory change | contemporaneous | 4.08 | 0.40 | 0.092 | ✗ |
| Crude inventory change | lag 1 (predictive) | 7.01 | 0.14 | 0.121 | ✗ |

The result sharpens the driver discussion. The dollar's strong *contemporaneous*
association (p ≈ 0.0005) all but disappears when it is lagged (p ≈ 0.65): the dollar
is a **coincident co-mover**, not a leading indicator, so — like Brent — it belongs to
the descriptive rather than the forecasting core. The VIX shows the opposite pattern:
insignificant contemporaneously (p ≈ 0.31) but **significant when lagged** (p ≈ 0.029,
Cramér's V ≈ 0.15), i.e. elevated equity-market stress this month tilts the odds toward
a *Down* WTI state next month. Lagged inventory change points the same way but is not
significant at 5%. Accordingly, the forecast-usable non-oil parent for the Bayesian
network is **lagged VIX**, with lagged inventories as a secondary candidate, while the
dollar enters the graph as a contemporaneous conditioning node alongside Brent.

*(Figure: `figures/fig_contemp_vs_lag.png` — Cramér's V by driver, contemporaneous vs
lag-1, with 5%-significant bars starred.)*
