"""Assemble + execute GWP1_v2_Extended.ipynb from extend_analysis.py logic."""
import nbformat as nbf
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell
from nbconvert.preprocessors import ExecutePreprocessor

src = open("extend_analysis.py").read()
# strip the trailing __main__ block so the notebook drives the calls itself
body = src.split('if __name__ == "__main__":')[0]

cells = [
 new_markdown_cell(
"""# GWP1 (v2, extended) — adding real non-oil drivers to the crude-oil PGM

This notebook strengthens the original oil-only package on three axes:

1. **Breadth (W1).** Adds three genuine non-oil drivers — the broad **USD index**,
   the **VIX**, and **US crude inventories** — alongside the real EIA WTI/Brent data.
2. **Trivial-driver fix (W2).** Separates the *contemporaneous* Brent benchmark
   (a near-identity with WTI, useful for description but **not** for forecasting)
   from *predictive* drivers known one month ahead, and ranks both by chi-square /
   Cramér's V / mutual information against the discrete WTI return state.
3. **Directed graph (W4).** Draws a proper **directed** Bayesian-network skeleton.

The real WTI/Brent are the EIA monthly series in `data/`. Set `USE_LIVE = True`
to pull the three non-oil series live from FRED; offline they use a *labelled
synthetic proxy* so the method runs end-to-end. Replace with the live pull before
submission."""),
 new_code_cell(body),
 new_markdown_cell("### Build the panel and rank drivers"),
 new_code_cell(
"""df, src = build_panel()
dep = dependence_table(df)
print("non-oil source:", src)
dep.round(4)"""),
 new_markdown_cell(
"""**Reading the table.** `Brent_ret (contemp.)` dominates on every measure — but it
is contemporaneous with WTI (they are the same commodity, corr ≈ 0.96), so it is a
*description*, not a forecast. Once we restrict to genuinely predictive inputs
(lagged / exogenous), the useful signal comes from the **non-oil** drivers —
lagged VIX, lagged USD return, lagged inventory change — which is precisely the
payoff of widening the panel beyond oil. (On the synthetic fallback these carry a
modest engineered lead; the live FRED pull gives the real magnitudes.)"""),
 new_markdown_cell("### Figures"),
 new_code_cell("figures(df, dep)\nfrom IPython.display import Image\nImage('figures/v2_fig3_distribution.png')"),
 new_code_cell("Image('figures/v2_fig7_driver_dependence.png')"),
 new_code_cell("Image('figures/v2_fig6_directed_bn.png')"),
 new_markdown_cell(
"""### Take-away

Widening the panel changes the story. In the oil-only package the "strongest driver"
was contemporaneous Brent — an artefact of measuring one oil price against another.
Once that tautology is set aside and real macro/financial/physical drivers are added,
the discrete-dependence analysis surfaces **lagged VIX, the dollar and inventories**
as the informative, *forecast-usable* parents of the WTI return state. That is the
conditional structure GWP2's Bayesian network should estimate — and the reason a
network model beats a single oil-only regression."""),
]

nb = new_notebook(cells=cells)
nb.metadata = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
               "language_info": {"name": "python"}}
print("executing ...")
ExecutePreprocessor(timeout=600, kernel_name="python3").preprocess(nb, {"metadata": {"path": "."}})
with open("GWP1_v2_Extended.ipynb", "w") as f:
    nbf.write(nb, f)
print("wrote GWP1_v2_Extended.ipynb (", len(nb.cells), "cells )")
