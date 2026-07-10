# GWP1 patch — lagged (predictive) dependence for the non-oil drivers

A small, drop-in addition to the **Improved GWP1 package**. That package screens the
non-oil drivers (dollar, VIX, inventories) *contemporaneously* only, so its dependence
table cannot separate same-month co-movement from genuine one-month-ahead
forecastability. This patch runs both screens on the package's own real panel and
resolves the distinction with actual numbers.

## The finding (real panel, 2006–2025, 239 months)

| Driver | contemporaneous | lag 1 (predictive) |
|---|---|---|
| Dollar index return | p = 0.0005 ✔ | p = 0.65 ✗ |
| VIX change | p = 0.31 ✗ | **p = 0.029 ✔** |
| Crude inventory change | p = 0.40 ✗ | p = 0.14 ✗ |

The dollar is a **coincident co-mover** (significant same-month, not when lagged) — like
Brent, descriptive not forecast-usable. **Lagged VIX** is the genuine predictive non-oil
signal the contemporaneous-only table misses. This becomes the forecast-usable parent for
the GWP2 Bayesian network.

## Files

| File | Use |
|---|---|
| `lagged_dependence_patch.py` | Standalone script: recomputes the table + figure. |
| `drop_in_notebook_cell.py` | Paste directly into the Improved notebook (uses `model_df` if present, else loads the panel). |
| `report_snippet.md` | Ready-to-paste report paragraph + Table 7b. |
| `dependence_contemp_vs_lag.csv` | Computed table. |
| `figures/fig_contemp_vs_lag.png` | Cramér's V by driver, contemporaneous vs lag-1 (significant bars starred). |
| `gwp1_augmented_panel_model.csv` | The Improved package's real panel (input). |

## Run

```bash
pip install numpy pandas scipy scikit-learn matplotlib
python3 lagged_dependence_patch.py
```
