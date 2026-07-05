"""
build_dataset.py
================
Reproducible construction of the GWP1 working dataset.

PRIMARY PATH (run locally / on Colab):
    Set USE_LIVE = True to pull the real series from FRED (CSV endpoint, no API
    key) and Yahoo Finance (yfinance). This is the data that belongs in the final
    submission.

FALLBACK PATH (used here, offline):
    Set USE_LIVE = False to regenerate a synthetic weekly panel (2005-2024) whose
    statistical fingerprint deliberately reproduces the documented stylized facts
    of crude oil (fat tails, volatility clustering, price spikes, inverse USD
    link). This lets the whole cleaning + EDA pipeline run end-to-end and produce
    real figures/statistics without network access. Every synthetic series is
    clearly flagged in the output files.

The variable list mirrors the Alvi (2018) macro / micro / financial / geopolitical
grouping used throughout the report.
"""

import json
import numpy as np
import pandas as pd

RNG_SEED = 20240628          # fixed for reproducibility
USE_LIVE = False             # <-- flip to True to download the real series
START = "2005-01-07"
END = "2024-05-31"
OUTDIR = "data"

# ----------------------------------------------------------------------------
# FRED / Yahoo identifiers for the live path (documented in the data dictionary)
# ----------------------------------------------------------------------------
FRED_SERIES = {
    "WTI":            "DCOILWTICO",      # WTI spot, USD/bbl (target)
    "Brent":          "DCOILBRENTEU",    # Brent spot, USD/bbl
    "USD_Index":      "DTWEXBGS",        # Broad trade-weighted USD index
    "UST10Y":         "DGS10",           # 10-year Treasury yield, %
    "CPI":            "CPIAUCSL",        # US CPI (monthly)
    "IndProd":        "INDPRO",          # US industrial production (monthly)
    "US_Production":  "MCRFPUS2",        # US field production of crude (monthly)
    "US_Inventory":   "WCESTUS1",        # US ending stocks of crude (weekly)
    "SP500":          "SP500",           # S&P 500 index
    "Gold":           "GOLDPMGBD228NLBM",# LBMA gold price, USD/oz
    "VIX":            "VIXCLS",          # CBOE VIX
    "OVX":            "OVXCLS",          # CBOE crude-oil volatility index
    "NatGas":         "DHHNGSP",         # Henry Hub natural gas spot
}
YF_TICKERS = {"XLE": "XLE"}             # Energy sector ETF (yfinance)
# Geopolitical Risk index: Caldara & Iacoviello, matteoiacoviello.com/gpr.htm


def _fetch_live() -> pd.DataFrame:
    """Download the real series. Requires internet + yfinance."""
    import io
    import requests
    import yfinance as yf

    frames = []
    for name, code in FRED_SERIES.items():
        url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={code}"
        txt = requests.get(url, timeout=30,
                           headers={"User-Agent": "Mozilla/5.0"}).text
        s = pd.read_csv(io.StringIO(txt), index_col=0, parse_dates=True)
        s.columns = [name]
        s[name] = pd.to_numeric(s[name], errors="coerce")
        frames.append(s)
    for name, tick in YF_TICKERS.items():
        px = yf.download(tick, start=START, end=END, progress=False)["Close"]
        px.name = name
        frames.append(px.to_frame())

    df = pd.concat(frames, axis=1).sort_index()
    df = df.resample("W-FRI").last()                 # weekly, Friday close
    df = df.loc[START:END]
    return df


# ----------------------------------------------------------------------------
# Synthetic generator (offline fallback)
# ----------------------------------------------------------------------------
def _simulate() -> pd.DataFrame:
    """
    Weekly panel that reproduces oil's stylized facts. The oil log-price follows
    a stochastic-volatility process with Poisson jumps, plus a handful of scripted
    macro regimes (2008 GFC, 2014-16 shale glut, 2020 COVID, 2022 invasion). Every
    other series is built from the same latent shocks so that the cross-series
    dependence structure (inverse USD, pro-cyclical equity, safe-haven gold,
    spiking VIX/OVX) is economically sensible.
    """
    rng = np.random.default_rng(RNG_SEED)
    idx = pd.date_range(START, END, freq="W-FRI")
    n = len(idx)
    t = np.arange(n)

    # --- latent global risk-appetite factor (drives co-movements) -----------
    risk = np.zeros(n)
    for i in range(1, n):
        risk[i] = 0.96 * risk[i - 1] + rng.normal(0, 1.0)

    # scripted crisis windows (index ranges) with extra risk-off pressure
    def win(d0, d1):
        return (idx >= d0) & (idx <= d1)
    crises = {
        "GFC":    (win("2008-07-01", "2009-03-31"), -6.0),
        "Shale":  (win("2014-07-01", "2016-02-28"), -3.5),
        "COVID":  (win("2020-02-20", "2020-05-15"), -9.0),
        "Ukraine":(win("2022-02-20", "2022-06-30"), +5.0),
    }
    shock = np.zeros(n)
    for mask, mag in crises.values():
        shock[mask] += mag

    # --- oil log-returns: SV + jumps + regime drift -------------------------
    vol = np.zeros(n)
    vol[0] = 0.040
    ret = np.zeros(n)
    for i in range(1, n):
        # GARCH-like variance recursion (volatility clustering), capped
        vol[i] = np.sqrt(0.00003 + 0.09 * ret[i - 1] ** 2 + 0.86 * vol[i - 1] ** 2)
        vol[i] = min(vol[i], 0.12)                     # keep the SV process stable
        drift = 0.0006 - 0.0009 * (risk[i] > 1.5)
        jump = 0.0
        if rng.random() < 0.025:                       # ~2.5% weekly jump prob
            # asymmetric jumps: downside jumps larger/more frequent (left skew)
            jump = rng.normal(-0.035, 0.09) if rng.random() < 0.60 \
                else rng.normal(0.025, 0.06)
        core = np.clip(rng.standard_t(df=5) / np.sqrt(5 / 3), -6, 6)  # bounded t
        ret[i] = drift + vol[i] * core + 0.010 * (shock[i] / 9.0) + jump
    ret[risk > 2.5] -= 0.01                            # deep risk-off drag
    # Re-centre to a realistic long-run drift so the 2005-2024 path round-trips
    # (rise into 2008, GFC crash, 2014-16 slide, 2020 collapse, 2022 spike)
    # instead of compounding to zero. Preserves clustering, jumps, crisis shapes.
    ret = ret - np.mean(ret) + 0.0007

    wti = 45.0 * np.exp(np.cumsum(ret))               # start ~ $45
    wti = np.clip(wti, 5, None)
    brent = wti * (1.03 + 0.02 * np.sin(t / 25) + rng.normal(0, 0.01, n))

    # --- macro / financial series built from shared shocks ------------------
    # USD index moves inversely to oil contemporaneously (dollar-denominated
    # commodity channel), with a slow independent drift on top.
    usd_ret = -0.18 * ret + 0.02 * (risk / 10) + rng.normal(0, 0.013, n)
    usd = 100 * np.exp(np.cumsum(usd_ret))
    ust10 = np.clip(3.0 - 0.04 * np.cumsum(rng.normal(0, 0.05, n))
                    - 0.15 * (risk / 10), 0.3, 6.0)
    sp_ret = 0.0012 + 0.20 * ret + 0.004 * (shock / 9) + rng.normal(0, 0.02, n)
    sp500 = 1200 * np.exp(np.cumsum(sp_ret))
    gold = 450 * np.exp(np.cumsum(0.0011 - 0.05 * sp_ret
                                  + 0.02 * (risk / 10) + rng.normal(0, 0.012, n)))
    vix = np.clip(17 - 6 * (sp_ret / 0.02) + 3 * np.abs(risk), 9, 85)
    ovx = np.clip(30 + 220 * np.abs(ret) + 2.0 * np.abs(risk), 12, 190)
    xle = 40 * np.exp(np.cumsum(0.0004 + 0.55 * ret + rng.normal(0, 0.015, n)))
    natgas = np.clip(6 * np.exp(np.cumsum(rng.normal(0, 0.05, n))), 1.3, 14)

    # monthly-frequency macro series (piecewise-constant within the month)
    months = idx.to_period("M")
    cpi = 190 * np.exp(0.0004 * t) * (1 + rng.normal(0, 0.001, n))
    indprod = 95 + 0.02 * t + 4 * np.sin(t / 26) - 0.3 * np.abs(shock)
    prod = 5.0 + 0.006 * t + 1.2 * (t > 470)           # shale ramp after ~2014
    prod += rng.normal(0, 0.05, n)
    inv = 350 + 25 * np.sin(t / 26 + 1) + np.cumsum(rng.normal(0, 0.8, n)) * 0.3

    # geopolitical risk index: baseline + spikes on crisis weeks
    gpr = 100 + 20 * np.abs(rng.normal(0, 1, n))
    for mask, _ in crises.values():
        gpr[mask] += rng.uniform(40, 120, mask.sum())

    df = pd.DataFrame(
        {
            "WTI": wti, "Brent": brent, "USD_Index": usd, "UST10Y": ust10,
            "CPI": cpi, "IndProd": indprod, "US_Production": prod,
            "US_Inventory": inv, "SP500": sp500, "Gold": gold, "VIX": vix,
            "OVX": ovx, "NatGas": natgas, "XLE": xle, "GPR": gpr,
        },
        index=idx,
    )
    df.index.name = "Date"

    # --- inject realistic data-quality problems for the cleaning step -------
    # (a) monthly series only reported once a month -> forward-fill gaps
    for col in ["CPI", "IndProd", "US_Production"]:
        keep = df.index.to_series().groupby(months).transform("first") == df.index.to_series()
        df.loc[~keep.values, col] = np.nan
    # (b) scattered missing prints
    for col in ["Brent", "NatGas", "Gold", "UST10Y"]:
        miss = rng.random(n) < 0.015
        df.loc[miss, col] = np.nan
    # (c) a few obvious bad ticks (fat-finger) and one duplicate row
    df.iloc[300, df.columns.get_loc("WTI")] *= 10       # 10x fat-finger
    df.iloc[620, df.columns.get_loc("USD_Index")] = 0.0  # impossible zero
    df.iloc[800, df.columns.get_loc("Gold")] = -5.0      # impossible negative
    return df


def build() -> pd.DataFrame:
    df = _fetch_live() if USE_LIVE else _simulate()
    df.to_csv(f"{OUTDIR}/oil_dataset_raw.csv")
    meta = {
        "source": "LIVE (FRED+Yahoo)" if USE_LIVE else "SYNTHETIC placeholder",
        "frequency": "weekly (W-FRI)",
        "n_rows": int(len(df)),
        "start": str(df.index.min().date()),
        "end": str(df.index.max().date()),
        "columns": list(df.columns),
        "seed": None if USE_LIVE else RNG_SEED,
    }
    with open(f"{OUTDIR}/dataset_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[build] wrote raw panel: {df.shape[0]} rows x {df.shape[1]} cols "
          f"({meta['source']})")
    return df


if __name__ == "__main__":
    build()
