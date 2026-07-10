"""
lagged_dependence_patch.py
==========================
Drop-in patch for the Improved GWP1 package. The improved package screens the
non-oil drivers CONTEMPORANEOUSLY only, so its dependence table conflates
same-month co-movement with genuine one-month-ahead forecastability. This script
runs BOTH the contemporaneous and the lag-1 (predictive) screen on the *real*
augmented panel and shows the difference that matters:

  - same-month dollar return is significant (co-movement) but LAGGED dollar is not
    -> not forecast-usable;
  - LAGGED VIX change IS significant -> a genuine, if weak, predictive driver the
    contemporaneous-only table misses.

Run:  python3 lagged_dependence_patch.py
Reproduces: dependence_contemp_vs_lag.csv and figures/fig_contemp_vs_lag.png
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.feature_selection import mutual_info_classif

PANEL = "gwp1_augmented_panel_model.csv"     # the improved package's real panel
SEED = 42


def tercile(s):
    q = s.quantile([1/3, 2/3])
    return pd.cut(s, [-np.inf, q.iloc[0], q.iloc[1], np.inf], labels=["Low", "Med", "High"])


def wti_state(r, band=0.01):
    return pd.cut(r, [-np.inf, -band, band, np.inf], labels=["Down", "Flat", "Up"])


def dep(y, x):
    """chi2, p, Cramér's V, mutual information for one driver vs the state."""
    d = pd.concat([y.rename("Y"), tercile(x).rename("X")], axis=1).dropna()
    ct = pd.crosstab(d["X"], d["Y"])
    chi2, p, _, _ = stats.chi2_contingency(ct)
    nobs = ct.values.sum()
    v = np.sqrt(chi2 / (nobs * (min(ct.shape) - 1)))
    mi = mutual_info_classif(pd.factorize(d["X"])[0].reshape(-1, 1),
                             pd.factorize(d["Y"])[0],
                             discrete_features=True, random_state=SEED)[0]
    return dict(chi2=chi2, p=p, cramers_v=v, mutual_info=mi, n=int(nobs))


def main():
    df = pd.read_csv(PANEL, parse_dates=["Date"]).set_index("Date")
    y = wti_state(df["WTI_ret"])

    # non-oil exogenous drivers: test contemporaneous AND one-month-lagged
    exo = {"Dollar index return": "DollarIndex_ret",
           "VIX change": "VIX_change",
           "Crude inventory change": "Inventory_change"}
    rows = []
    for label, col in exo.items():
        c = dep(y, df[col])
        l = dep(y, df[col].shift(1))
        rows.append([label, "contemporaneous", c["chi2"], c["p"], c["cramers_v"], c["mutual_info"], c["n"]])
        rows.append([label + " (lag 1)", "predictive", l["chi2"], l["p"], l["cramers_v"], l["mutual_info"], l["n"]])

    tab = pd.DataFrame(rows, columns=["driver", "type", "chi2", "p_value",
                                      "cramers_v", "mutual_info", "n"])
    tab["significant_5pct"] = tab["p_value"] < 0.05
    tab.to_csv("dependence_contemp_vs_lag.csv", index=False)

    # figure: contemporaneous vs lag-1 Cramér's V, grouped by driver -----------
    labels = list(exo.keys())
    contemp = [tab[(tab.driver == l)].cramers_v.values[0] for l in labels]
    lagged = [tab[(tab.driver == l + " (lag 1)")].cramers_v.values[0] for l in labels]
    x = np.arange(len(labels)); w = 0.38
    fig, ax = plt.subplots(figsize=(9, 5))
    b1 = ax.bar(x - w/2, contemp, w, label="contemporaneous (co-movement)", color="#b04a4a")
    b2 = ax.bar(x + w/2, lagged, w, label="lag 1 (predictive)", color="#2a7f4f")
    ax.axhline(0, color="k", lw=.6)
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=12)
    ax.set_ylabel("Cramér's V with WTI return state")
    ax.set_title("Non-oil drivers: same-month co-movement vs one-month-ahead prediction")
    # star the 5%-significant bars
    for bars, kind in [(b1, "contemporaneous"), (b2, "predictive")]:
        for rect, lab in zip(bars, labels):
            key = lab if kind == "contemporaneous" else lab + " (lag 1)"
            sig = tab[tab.driver == key].significant_5pct.values[0]
            if sig:
                ax.text(rect.get_x() + rect.get_width()/2, rect.get_height() + .005,
                        "*", ha="center", fontsize=16, color="black")
    ax.legend()
    fig.tight_layout(); fig.savefig("figures/fig_contemp_vs_lag.png", dpi=140)

    pd.set_option("display.width", 120)
    print("Real panel:", df.index.min().date(), "->", df.index.max().date(),
          f"({len(df)} months)\n")
    print(tab.to_string(index=False,
          formatters={"chi2": "{:.2f}".format, "p_value": "{:.4g}".format,
                      "cramers_v": "{:.3f}".format, "mutual_info": "{:.3f}".format}))
    print("\n(* = significant at 5%. Note: contemporaneous dollar is significant but "
          "its LAG is not -> co-movement, not a forecast; lagged VIX is the genuine "
          "predictive signal.)")


if __name__ == "__main__":
    main()
