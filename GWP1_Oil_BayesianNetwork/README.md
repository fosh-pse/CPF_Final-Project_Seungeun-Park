# GWP1 — Forecasting Crude Oil Prices with Probabilistic Graphical Models

MScFE 660 Risk Management · Group Work Project #1 · Seungeun Park

Empirical groundwork for a Bayesian-network model of the crude-oil price: problem
formulation, a 15-variable macro / industry / financial / geopolitical panel, a
three-stage cleaning protocol, exploratory data analysis, and the stylized-fact
discussion that motivates the methodology of GWP2.

## Contents

| File | What it is |
|---|---|
| `GWP1_Report.md` | The written report (paste into the WQU report template → export PDF). |
| `GWP1_Oil_BayesianNetwork.ipynb` | Executed notebook: data collection, cleaning, EDA, Step-8 answers. |
| `GWP1_Oil_BayesianNetwork.html` | HTML copy of the executed notebook (submission requirement). |
| `build_dataset.py` | Data layer — live FRED/Yahoo download **or** reproducible synthetic fallback. |
| `clean_and_eda.py` | Cleaning protocol + EDA figures + stylized-fact statistics. |
| `build_notebook.py` | Assembles and executes the notebook from source cells. |
| `data/` | `*_raw.csv`, `*_clean.csv`, `cleaning_log.txt`, `stylized_stats.json`, `dataset_meta.json`. |
| `figures/` | A–E EDA figures (PNG). |

## How to run

```bash
pip install numpy pandas matplotlib seaborn scipy statsmodels networkx
#           (+ requests yfinance nbformat nbconvert for the live pull / notebook build)

python3 build_dataset.py     # writes data/oil_dataset_raw.csv
python3 clean_and_eda.py     # writes cleaned data, figures/, stylized_stats.json
python3 build_notebook.py    # (re)builds + executes the notebook
```

## Live vs. synthetic data

The pipeline defaults to a **reproducible synthetic panel** (seed `20240628`) engineered to
reproduce oil's stylized facts, so everything runs offline and every figure is real.

**For the final submission, switch to the live data:** set `USE_LIVE = True` in
`build_dataset.py` (and the first cell of the notebook) and re-run top-to-bottom on a
machine with internet. FRED series come from the public CSV endpoint (no API key); the XLE
ETF via `yfinance`; the Geopolitical Risk index from Caldara & Iacoviello
(matteoiacoviello.com/gpr.htm). The qualitative conclusions are unchanged; the specific
statistics quoted in the report refresh automatically.
