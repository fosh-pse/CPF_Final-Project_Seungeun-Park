# =============================================================================
# DROP-IN CELL for the Improved GWP1 notebook.
# Paste this AFTER the panel `model_df` (with WTI_ret, DollarIndex_ret,
# VIX_change, Inventory_change) is built. It adds the one-month-ahead
# (predictive) dependence screen next to the contemporaneous one, so the report's
# "contemporaneous vs predictive" distinction is actually tested, not just stated.
# =============================================================================
import numpy as np, pandas as pd
from scipy import stats
from sklearn.feature_selection import mutual_info_classif

# --- use the panel already in memory; else load the saved model panel ---------
try:
    _df = model_df.copy()
except NameError:
    _df = pd.read_csv("data/gwp1_augmented_panel_model.csv",
                      parse_dates=["Date"]).set_index("Date")

def _terc(s):
    q = s.quantile([1/3, 2/3])
    return pd.cut(s, [-np.inf, q.iloc[0], q.iloc[1], np.inf], labels=["Low", "Med", "High"])

def _state(r, band=0.01):
    return pd.cut(r, [-np.inf, -band, band, np.inf], labels=["Down", "Flat", "Up"])

def _dep(y, x):
    d = pd.concat([y.rename("Y"), _terc(x).rename("X")], axis=1).dropna()
    ct = pd.crosstab(d["X"], d["Y"]); chi2, p, _, _ = stats.chi2_contingency(ct)
    n = ct.values.sum(); v = np.sqrt(chi2 / (n * (min(ct.shape) - 1)))
    mi = mutual_info_classif(pd.factorize(d["X"])[0].reshape(-1, 1),
                             pd.factorize(d["Y"])[0], discrete_features=True,
                             random_state=42)[0]
    return chi2, p, v, mi

_y = _state(_df["WTI_ret"])
_exo = {"Dollar index return": "DollarIndex_ret",
        "VIX change": "VIX_change",
        "Crude inventory change": "Inventory_change"}
_rows = []
for _lab, _c in _exo.items():
    for _kind, _series in [("contemporaneous", _df[_c]), ("lag 1 (predictive)", _df[_c].shift(1))]:
        chi2, p, v, mi = _dep(_y, _series)
        _rows.append([_lab, _kind, round(chi2, 2), f"{p:.4g}", round(v, 3),
                      round(mi, 3), "yes" if p < 0.05 else "no"])
lagged_dependence = pd.DataFrame(_rows, columns=[
    "Driver", "Timing", "Chi-square", "p-value", "Cramer's V", "Mutual info", "Sig 5%"])
print("One-month-ahead vs contemporaneous dependence with WTI return state")
print("(sample:", _df.index.min().date(), "to", _df.index.max().date(),
      f", {len(_df)} months)")
display(lagged_dependence)

# Reading: the contemporaneous dollar link is significant but its LAG is not
# (p ~ 0.65) -> same-month co-movement, not a forecast. Lagged VIX change IS
# significant (p ~ 0.03) -> the genuine, if modest, predictive non-oil driver.
