"""
clean_and_eda.py
================
Consumes data/oil_dataset_raw.csv, applies the three-stage cleaning protocol
(outliers / bad data / missing values) described in the report, then runs the
exploratory data analysis and writes every figure + a machine-readable summary
of the stylized-fact statistics that the report cites.

Run after build_dataset.py.
"""

import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from statsmodels.tsa.stattools import adfuller, acf
from statsmodels.stats.diagnostic import acorr_ljungbox

sns.set_theme(style="whitegrid", context="talk")
FIG = "figures"
DATA = "data"
TARGET = "WTI"


# ---------------------------------------------------------------------------
# 1. CLEANING  (Student A: outliers | B: bad data | C: missing values)
# ---------------------------------------------------------------------------
def clean(df: pd.DataFrame):
    log = []                                    # audit trail of every removal

    # --- B: bad / impossible data -----------------------------------------
    price_like = ["WTI", "Brent", "USD_Index", "CPI", "IndProd",
                  "US_Production", "US_Inventory", "SP500", "Gold",
                  "NatGas", "XLE", "GPR"]
    for c in price_like:
        bad = df[c] <= 0
        if bad.any():
            log.append(f"[bad-data] {c}: {int(bad.sum())} non-positive value(s) "
                       f"set to NaN -> {list(df.index[bad].date.astype(str))}")
            df.loc[bad, c] = np.nan
    dups = df.index.duplicated().sum()
    if dups:
        log.append(f"[bad-data] dropped {dups} duplicated timestamp row(s)")
        df = df[~df.index.duplicated()]

    # --- A: extreme outliers (robust z on log-returns of tradable prices) --
    for c in ["WTI", "Brent", "SP500", "Gold", "XLE", "NatGas"]:
        r = np.log(df[c]).diff()
        med, mad = r.median(), (r - r.median()).abs().median()
        rz = 0.6745 * (r - med) / (mad if mad else np.nan)   # modified z-score
        ext = rz.abs() > 8                                   # |z|>8 = fat-finger
        if ext.any():
            log.append(f"[outlier] {c}: {int(ext.sum())} print(s) with |robust z|>8 "
                       f"on returns flagged -> {list(df.index[ext].date.astype(str))}")
            df.loc[ext, c] = np.nan                          # hand to imputation

    # --- C: missing values -------------------------------------------------
    # monthly macro series: forward-fill (value valid until next release)
    for c in ["CPI", "IndProd", "US_Production"]:
        miss = int(df[c].isna().sum())
        df[c] = df[c].ffill().bfill()
        log.append(f"[missing] {c}: {miss} gaps forward/back-filled (release-date carry)")
    # market series: time interpolation for short gaps, then edge-fill
    for c in ["Brent", "NatGas", "Gold", "UST10Y", "US_Inventory",
              "WTI", "SP500", "XLE", "USD_Index"]:
        miss = int(df[c].isna().sum())
        if miss:
            df[c] = df[c].interpolate(method="time").ffill().bfill()
            log.append(f"[missing] {c}: {miss} gap(s) linearly interpolated in time")

    assert df.isna().sum().sum() == 0, "cleaning left residual NaNs"
    df.to_csv(f"{DATA}/oil_dataset_clean.csv")
    with open(f"{DATA}/cleaning_log.txt", "w") as f:
        f.write("\n".join(log))
    print(f"[clean] sterilized panel written; {len(log)} cleaning actions logged")
    return df, log


# ---------------------------------------------------------------------------
# 2. STYLIZED-FACT STATISTICS
# ---------------------------------------------------------------------------
def stylized_stats(df: pd.DataFrame) -> dict:
    r = np.log(df[TARGET]).diff().dropna()
    jb_stat, jb_p = stats.jarque_bera(r)
    adf_lvl = adfuller(df[TARGET].dropna(), autolag="AIC")
    adf_ret = adfuller(r, autolag="AIC")
    lb_r = acorr_ljungbox(r, lags=[10], return_df=True)
    lb_r2 = acorr_ljungbox(r ** 2, lags=[10], return_df=True)
    out = {
        "n_returns": int(r.size),
        "mean_weekly_return": float(r.mean()),
        "ann_vol": float(r.std() * np.sqrt(52)),
        "skewness": float(stats.skew(r)),
        "excess_kurtosis": float(stats.kurtosis(r)),          # 0 => normal
        "jarque_bera_stat": float(jb_stat),
        "jarque_bera_p": float(jb_p),
        "adf_level_stat": float(adf_lvl[0]),
        "adf_level_p": float(adf_lvl[1]),
        "adf_return_stat": float(adf_ret[0]),
        "adf_return_p": float(adf_ret[1]),
        "ljungbox_ret_p": float(lb_r["lb_pvalue"].iloc[0]),
        "ljungbox_ret2_p": float(lb_r2["lb_pvalue"].iloc[0]),
        "corr_WTI_USD": float(df["WTI"].pct_change().corr(df["USD_Index"].pct_change())),
        "corr_WTI_SP500": float(df["WTI"].pct_change().corr(df["SP500"].pct_change())),
        "corr_WTI_XLE": float(df["WTI"].pct_change().corr(df["XLE"].pct_change())),
    }
    with open(f"{DATA}/stylized_stats.json", "w") as f:
        json.dump(out, f, indent=2)
    print("[stats] stylized-fact statistics written")
    return out


# ---------------------------------------------------------------------------
# 3. FIGURES
# ---------------------------------------------------------------------------
def figures(df: pd.DataFrame):
    r = np.log(df[TARGET]).diff().dropna()

    # (A) DISTRIBUTIONAL ----------------------------------------------------
    fig, ax = plt.subplots(1, 2, figsize=(15, 5.5))
    sns.histplot(r, bins=80, stat="density", kde=True, ax=ax[0], color="#3b6ea5")
    x = np.linspace(r.min(), r.max(), 400)
    ax[0].plot(x, stats.norm.pdf(x, r.mean(), r.std()), "r--", lw=2,
               label="Normal fit")
    ax[0].set(title=f"(A1) {TARGET} weekly log-returns vs Normal",
              xlabel="log-return")
    ax[0].legend()
    stats.probplot(r, dist="norm", plot=ax[1])
    ax[1].get_lines()[0].set(marker="o", markersize=3, alpha=.5, color="#3b6ea5")
    ax[1].get_lines()[1].set(color="r", lw=2)
    ax[1].set_title("(A2) Normal Q-Q plot (fat tails bend off the line)")
    fig.tight_layout(); fig.savefig(f"{FIG}/A_distributional.png", dpi=130)
    plt.close(fig)

    # (B) TIME SERIES -------------------------------------------------------
    fig, ax = plt.subplots(3, 1, figsize=(15, 12), sharex=True)
    ax[0].plot(df.index, df[TARGET], color="#1a1a1a", lw=1)
    ax[0].set(title=f"(B1) {TARGET} spot price with event annotations",
              ylabel="USD/bbl")
    for d, lab in [("2008-09-15", "GFC"), ("2014-11-27", "OPEC/shale glut"),
                   ("2020-04-20", "COVID"), ("2022-02-24", "Ukraine")]:
        ax[0].axvline(pd.Timestamp(d), color="crimson", ls=":", lw=1.2)
        ax[0].text(pd.Timestamp(d), df[TARGET].max() * .9, lab, rotation=90,
                   fontsize=10, color="crimson", va="top")
    ax[1].plot(r.index, r, color="#3b6ea5", lw=.7)
    ax[1].set(title="(B2) Weekly log-returns — note the clustered turbulence",
              ylabel="log-return")
    rv = r.rolling(13).std() * np.sqrt(52)
    ax[2].plot(rv.index, rv, color="#b5651d", lw=1.3)
    ax[2].set(title="(B3) 13-week rolling annualised volatility (clustering)",
              ylabel="ann. vol")
    fig.tight_layout(); fig.savefig(f"{FIG}/B_timeseries.png", dpi=130)
    plt.close(fig)

    # (C) MULTIVARIATE ------------------------------------------------------
    rets = df[["WTI", "Brent", "USD_Index", "SP500", "Gold", "XLE",
               "NatGas"]].pct_change().dropna()
    lvl = df[["UST10Y", "VIX", "OVX", "US_Production",
              "US_Inventory", "GPR"]]
    corr = pd.concat([rets, lvl.pct_change().add_suffix("_chg")],
                     axis=1, sort=False).dropna().corr()
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0,
                square=True, cbar_kws={"shrink": .8}, ax=ax, annot_kws={"size": 8})
    ax.set_title("(C1) Correlation of weekly changes across the panel")
    fig.tight_layout(); fig.savefig(f"{FIG}/C_correlation.png", dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(1, 2, figsize=(15, 5.5))
    ax[0].scatter(df["USD_Index"].pct_change(), df["WTI"].pct_change(),
                  s=8, alpha=.4, color="#3b6ea5")
    ax[0].set(title="(C2) WTI vs USD weekly change (inverse link)",
              xlabel="USD index change", ylabel="WTI change")
    roll = df["WTI"].pct_change().rolling(52).corr(df["USD_Index"].pct_change())
    ax[1].plot(roll.index, roll, color="#b5651d")
    ax[1].axhline(0, color="k", lw=.8)
    ax[1].set(title="(C3) 52-week rolling corr(WTI, USD) — time-varying",
              ylabel="rolling corr")
    fig.tight_layout(); fig.savefig(f"{FIG}/C_multivariate.png", dpi=130)
    plt.close(fig)

    # (D) AUTOCORRELATION of returns vs squared returns ---------------------
    lags = 26
    ac_r = acf(r, nlags=lags, fft=True)[1:]
    ac_r2 = acf(r ** 2, nlags=lags, fft=True)[1:]
    ci = 1.96 / np.sqrt(len(r))
    fig, ax = plt.subplots(1, 2, figsize=(15, 5.5), sharey=True)
    ax[0].bar(range(1, lags + 1), ac_r, color="#3b6ea5")
    ax[0].axhline(ci, color="r", ls="--"); ax[0].axhline(-ci, color="r", ls="--")
    ax[0].set(title="(D1) ACF of returns (≈ white noise)", xlabel="lag")
    ax[1].bar(range(1, lags + 1), ac_r2, color="#b5651d")
    ax[1].axhline(ci, color="r", ls="--"); ax[1].axhline(-ci, color="r", ls="--")
    ax[1].set(title="(D2) ACF of squared returns (long memory in vol)",
              xlabel="lag")
    fig.tight_layout(); fig.savefig(f"{FIG}/D_autocorrelation.png", dpi=130)
    plt.close(fig)

    # (E) proposed Bayesian-network skeleton (qualitative, for GWP2 bridge) --
    import networkx as nx
    G = nx.DiGraph()
    edges = [("GPR", "WTI"), ("USD_Index", "WTI"), ("US_Inventory", "WTI"),
             ("US_Production", "US_Inventory"), ("IndProd", "WTI"),
             ("WTI", "OVX"), ("WTI", "XLE"), ("WTI", "CPI"),
             ("UST10Y", "USD_Index"), ("SP500", "WTI"), ("VIX", "SP500")]
    G.add_edges_from(edges)
    pos = nx.spring_layout(G, seed=RNG if (RNG := 7) else 7, k=1.1)
    fig, ax = plt.subplots(figsize=(11, 8))
    nx.draw_networkx(G, pos, ax=ax, node_color="#cfe0f2", node_size=2200,
                     font_size=10, edge_color="#666", arrows=True,
                     arrowsize=18, width=1.4)
    ax.set_title("(E) Candidate causal skeleton for the oil-price Bayesian network")
    ax.axis("off")
    fig.tight_layout(); fig.savefig(f"{FIG}/E_bn_skeleton.png", dpi=130)
    plt.close(fig)

    print("[figures] 6 figure files written to figures/")


if __name__ == "__main__":
    raw = pd.read_csv(f"{DATA}/oil_dataset_raw.csv", index_col=0, parse_dates=True)
    clean_df, _ = clean(raw.copy())
    stats_out = stylized_stats(clean_df)
    figures(clean_df)
    print("\n=== stylized-fact summary ===")
    print(json.dumps(stats_out, indent=2))
