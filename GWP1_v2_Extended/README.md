# GWP1 v2 — Extended (real oil data + non-oil FRED drivers)

An improved version of the uploaded `GWP1_Crude_Oil_PGM_Final_Package`, keeping its **real
EIA WTI/Brent monthly data** and fixing three weaknesses:

1. **Breadth** — adds three real non-oil drivers (USD index `DTWEXBGS`, VIX `VIXCLS`,
   US crude inventories `WCESTUS1`).
2. **Trivial driver** — separates contemporaneous Brent (non-predictive) from genuinely
   predictive lagged/exogenous drivers, ranked by χ² / Cramér's V / mutual information.
3. **Graph** — proper directed Bayesian-network skeleton; fixed distribution figure.

## Contents

| File | What |
|---|---|
| `extend_analysis.py` | Panel build + discrete-dependence analysis + figures. |
| `build_v2_notebook.py` | Assembles/executes the notebook. |
| `GWP1_v2_Extended.ipynb` / `.html` | Executed notebook. |
| `GWP1_v2_ADDENDUM.md` | Report sections to paste into the main report. |
| `data/wti-monthly.csv`, `brent-monthly.csv` | Real EIA monthly prices (from the uploaded package). |
| `data/v2_driver_dependence.csv`, `v2_summary.json` | Computed results. |
| `figures/v2_fig3_distribution.png`, `v2_fig6_directed_bn.png`, `v2_fig7_driver_dependence.png` | Figures. |

## Run

```bash
pip install numpy pandas matplotlib scipy scikit-learn networkx nbformat nbconvert
python3 extend_analysis.py      # -> figures/, data/v2_*.csv/json
python3 build_v2_notebook.py    # -> executed notebook (+ nbconvert for HTML)
```

## Live vs synthetic

Real WTI/Brent are used as-is. The three **non-oil** series are downloaded live from FRED
when `USE_LIVE = True`; offline they use a **labelled synthetic proxy** (with a small
economically-motivated lead on next-month oil) so the pipeline and the
contemporaneous-vs-predictive method run end-to-end. **Switch to `USE_LIVE = True` and
re-run before submission** to replace the non-oil numbers with the real FRED values.
