"""
build_notebook.py
=================
Assembles GWP1_Oil_BayesianNetwork.ipynb from source cells, then executes it so
that every figure and printed output is embedded in the committed notebook.
Run:  python3 build_notebook.py
"""
import nbformat as nbf
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell
from nbconvert.preprocessors import ExecutePreprocessor

md = new_markdown_cell
code = new_code_cell
cells = []

# ---------------------------------------------------------------- title -----
cells.append(md(
"""# Group Work Project #1 — Data Collection & Exploration
## Forecasting Crude Oil Prices with Probabilistic Graphical Models

**MScFE 660 Risk Management — WorldQuant University**

This notebook accompanies the GWP1 report. It (1) collects the macroeconomic,
industry, financial and geopolitical drivers of the crude-oil price, (2) applies
a three-stage cleaning protocol, and (3) runs the exploratory data analysis that
grounds the stylized-fact discussion (report Step 8).

> **Data note.** Set `USE_LIVE = True` to pull the real series from FRED and Yahoo
> Finance — that is the data used for the final submission. When run without
> internet (`USE_LIVE = False`) the notebook regenerates a *reproducible synthetic
> panel* whose statistical fingerprint reproduces oil's documented stylized facts,
> so the pipeline and every figure run end-to-end. Synthetic output is labelled as
> such and must be replaced by the live pull before submission."""))

# ---------------------------------------------------------------- imports ----
cells.append(md("### 1 · Setup"))
cells.append(code(
"""import json, io, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from statsmodels.tsa.stattools import adfuller, acf
from statsmodels.stats.diagnostic import acorr_ljungbox

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", context="notebook")
pd.set_option("display.width", 120)

USE_LIVE = False          # <-- True to download the real FRED/Yahoo series
SEED     = 20240628
START, END = "2005-01-07", "2024-05-31"
TARGET   = "WTI"
print("USE_LIVE =", USE_LIVE)"""))

# --------------------------------------------------- data dictionary ---------
cells.append(md(
"""### 2 · Data dictionary

Variables are grouped following Alvi (2018): **macroeconomic / geopolitical**,
**microeconomic (industry)** and **financial**. All series are aligned to a weekly
(Friday) grid over 2005–2024."""))
cells.append(code(
'''FRED = {   # series id, group, unit, native frequency
 "WTI":("DCOILWTICO","target","USD/bbl","daily"),
 "Brent":("DCOILBRENTEU","target","USD/bbl","daily"),
 "USD_Index":("DTWEXBGS","macro","index","daily"),
 "UST10Y":("DGS10","macro","%","daily"),
 "CPI":("CPIAUCSL","macro","index","monthly"),
 "IndProd":("INDPRO","macro","index","monthly"),
 "US_Production":("MCRFPUS2","micro","kbbl/d","monthly"),
 "US_Inventory":("WCESTUS1","micro","kbbl","weekly"),
 "SP500":("SP500","financial","index","daily"),
 "Gold":("GOLDPMGBD228NLBM","financial","USD/oz","daily"),
 "VIX":("VIXCLS","financial","index","daily"),
 "OVX":("OVXCLS","financial","index","daily"),
 "NatGas":("DHHNGSP","financial","USD/MMBtu","daily"),
}
YF = {"XLE":("XLE","financial","USD","daily")}   # energy-sector ETF (Yahoo)
# Geopolitical Risk index (GPR): Caldara & Iacoviello, matteoiacoviello.com
dd = pd.DataFrame(
    [(k,)+v for k,v in {**FRED, **YF}.items()],
    columns=["variable","source_id","group","unit","native_freq"]
).set_index("variable")
dd.loc["GPR"] = ["GPR (C&I)","geopolitical","index","monthly"]
dd'''))

# ---------------------------------------------------- data collection --------
cells.append(md("### 3 · Data collection\n\n**Live path** (submission) and **synthetic fallback** (offline)."))
cells.append(code(
'''def fetch_live():
    """Download real series from FRED (CSV endpoint) + Yahoo (yfinance)."""
    import requests, yfinance as yf
    frames = []
    for name,(code_,*_ ) in FRED.items():
        url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={code_}"
        txt = requests.get(url, timeout=30, headers={"User-Agent":"Mozilla/5.0"}).text
        s = pd.read_csv(io.StringIO(txt), index_col=0, parse_dates=True)
        s.columns=[name]; s[name]=pd.to_numeric(s[name], errors="coerce")
        frames.append(s)
    for name,(tick,*_ ) in YF.items():
        px = yf.download(tick, start=START, end=END, progress=False)["Close"]
        px.name=name; frames.append(px.to_frame())
    df = pd.concat(frames, axis=1).sort_index().resample("W-FRI").last().loc[START:END]
    return df'''))
cells.append(code(open("_nb_simulate.py").read() if False else
'''def simulate():
    """Weekly panel reproducing oil stylized facts (SV + jumps + crisis regimes)."""
    rng = np.random.default_rng(SEED)
    idx = pd.date_range(START, END, freq="W-FRI"); n=len(idx); t=np.arange(n)
    risk=np.zeros(n)
    for i in range(1,n): risk[i]=0.96*risk[i-1]+rng.normal(0,1.0)
    def win(a,b): return (idx>=a)&(idx<=b)
    crises={ "GFC":(win("2008-07-01","2009-03-31"),-6.0),
             "Shale":(win("2014-07-01","2016-02-28"),-3.5),
             "COVID":(win("2020-02-20","2020-05-15"),-9.0),
             "Ukraine":(win("2022-02-20","2022-06-30"),+5.0)}
    shock=np.zeros(n)
    for m,mag in crises.values(): shock[m]+=mag
    vol=np.zeros(n); vol[0]=0.04; ret=np.zeros(n)
    for i in range(1,n):
        vol[i]=min(np.sqrt(0.00003+0.09*ret[i-1]**2+0.86*vol[i-1]**2),0.12)
        drift=0.0006-0.0009*(risk[i]>1.5); jump=0.0
        if rng.random()<0.025:
            jump=rng.normal(-0.035,0.09) if rng.random()<0.60 else rng.normal(0.025,0.06)
        core=np.clip(rng.standard_t(df=5)/np.sqrt(5/3),-6,6)
        ret[i]=drift+vol[i]*core+0.010*(shock[i]/9.0)+jump
    ret[risk>2.5]-=0.01
    ret=ret-np.mean(ret)+0.0007                     # realistic long-run drift
    wti=np.clip(45.0*np.exp(np.cumsum(ret)),5,None)
    brent=wti*(1.03+0.02*np.sin(t/25)+rng.normal(0,0.01,n))
    usd=100*np.exp(np.cumsum(-0.18*ret+0.02*(risk/10)+rng.normal(0,0.013,n)))
    ust10=np.clip(3.0-0.04*np.cumsum(rng.normal(0,0.05,n))-0.15*(risk/10),0.3,6.0)
    spr=0.0012+0.20*ret+0.004*(shock/9)+rng.normal(0,0.02,n); sp500=1200*np.exp(np.cumsum(spr))
    gold=450*np.exp(np.cumsum(0.0011-0.05*spr+0.02*(risk/10)+rng.normal(0,0.012,n)))
    vix=np.clip(17-6*(spr/0.02)+3*np.abs(risk),9,85)
    ovx=np.clip(30+220*np.abs(ret)+2.0*np.abs(risk),12,190)
    xle=40*np.exp(np.cumsum(0.0004+0.55*ret+rng.normal(0,0.015,n)))
    natgas=np.clip(6*np.exp(np.cumsum(rng.normal(0,0.05,n))),1.3,14)
    months=idx.to_period("M")
    cpi=190*np.exp(0.0004*t)*(1+rng.normal(0,0.001,n))
    indprod=95+0.02*t+4*np.sin(t/26)-0.3*np.abs(shock)
    prod=5.0+0.006*t+1.2*(t>470)+rng.normal(0,0.05,n)
    inv=350+25*np.sin(t/26+1)+np.cumsum(rng.normal(0,0.8,n))*0.3
    gpr=100+20*np.abs(rng.normal(0,1,n))
    for m,_ in crises.values(): gpr[m]+=rng.uniform(40,120,m.sum())
    df=pd.DataFrame({"WTI":wti,"Brent":brent,"USD_Index":usd,"UST10Y":ust10,"CPI":cpi,
        "IndProd":indprod,"US_Production":prod,"US_Inventory":inv,"SP500":sp500,
        "Gold":gold,"VIX":vix,"OVX":ovx,"NatGas":natgas,"XLE":xle,"GPR":gpr},index=idx)
    df.index.name="Date"
    # inject data-quality problems for the cleaning step -----------------------
    for c in ["CPI","IndProd","US_Production"]:
        keep=df.index.to_series().groupby(months).transform("first")==df.index.to_series()
        df.loc[~keep.values,c]=np.nan
    for c in ["Brent","NatGas","Gold","UST10Y"]:
        df.loc[rng.random(n)<0.015,c]=np.nan
    df.iloc[300,df.columns.get_loc("WTI")]*=10       # 10x fat-finger
    df.iloc[620,df.columns.get_loc("USD_Index")]=0.0 # impossible zero
    df.iloc[800,df.columns.get_loc("Gold")]=-5.0     # impossible negative
    return df

raw = fetch_live() if USE_LIVE else simulate()
print(("LIVE" if USE_LIVE else "SYNTHETIC"), "panel:", raw.shape)
raw.tail(3)'''))

# ---------------------------------------------------------- cleaning ---------
cells.append(md(
"""### 4 · Cleaning — three-stage protocol

Following report Step 5 the cleaning is split by responsibility:

* **Bad data** — impossible values (non-positive prices/indices) → `NaN`; duplicate timestamps dropped.
* **Extreme outliers** — a modified (MAD-based) z-score on log-returns flags fat-finger prints (`|z| > 8`).
* **Missing values** — monthly macro series carried forward (release-date logic); market gaps interpolated in time.

Every action is written to an auditable cleaning log (report Step 6)."""))
cells.append(code(
'''def clean(df):
    log=[]
    price_like=["WTI","Brent","USD_Index","CPI","IndProd","US_Production",
                "US_Inventory","SP500","Gold","NatGas","XLE","GPR"]
    for c in price_like:                              # --- bad data ---
        bad=df[c]<=0
        if bad.any():
            log.append(f"[bad-data] {c}: {int(bad.sum())} non-positive -> NaN "
                       f"{list(df.index[bad].date.astype(str))}")
            df.loc[bad,c]=np.nan
    d=df.index.duplicated().sum()
    if d: log.append(f"[bad-data] dropped {d} duplicate row(s)"); df=df[~df.index.duplicated()]
    for c in ["WTI","Brent","SP500","Gold","XLE","NatGas"]:   # --- outliers ---
        r=np.log(df[c]).diff(); med,mad=r.median(),(r-r.median()).abs().median()
        rz=0.6745*(r-med)/(mad if mad else np.nan); ext=rz.abs()>8
        if ext.any():
            log.append(f"[outlier] {c}: {int(ext.sum())} print(s) |robust z|>8 "
                       f"{list(df.index[ext].date.astype(str))}"); df.loc[ext,c]=np.nan
    for c in ["CPI","IndProd","US_Production"]:               # --- missing ---
        m=int(df[c].isna().sum()); df[c]=df[c].ffill().bfill()
        log.append(f"[missing] {c}: {m} gaps carried forward (release-date)")
    for c in ["Brent","NatGas","Gold","UST10Y","US_Inventory","WTI","SP500","XLE","USD_Index"]:
        m=int(df[c].isna().sum())
        if m: df[c]=df[c].interpolate(method="time").ffill().bfill()
        if m: log.append(f"[missing] {c}: {m} gap(s) time-interpolated")
    assert df.isna().sum().sum()==0
    return df, log

clean_df, cleaning_log = clean(raw.copy())
print("\\n".join(cleaning_log))'''))

# ------------------------------------------------------ stylized stats -------
cells.append(md("### 5 · Stylized-fact statistics (report Step 8)"))
cells.append(code(
'''r = np.log(clean_df[TARGET]).diff().dropna()
jb  = stats.jarque_bera(r)
adf_l = adfuller(clean_df[TARGET], autolag="AIC")
adf_r = adfuller(r, autolag="AIC")
lb_r  = acorr_ljungbox(r,   lags=[10], return_df=True)["lb_pvalue"].iloc[0]
lb_r2 = acorr_ljungbox(r**2,lags=[10], return_df=True)["lb_pvalue"].iloc[0]
summary = pd.Series({
 "n_returns":r.size,
 "mean weekly return":r.mean(),
 "annualised vol":r.std()*np.sqrt(52),
 "skewness":stats.skew(r),
 "excess kurtosis":stats.kurtosis(r),
 "Jarque-Bera p":jb.pvalue,
 "ADF level p (price)":adf_l[1],
 "ADF p (returns)":adf_r[1],
 "Ljung-Box p, returns":lb_r,
 "Ljung-Box p, squared returns":lb_r2,
 "corr(WTI,USD)":clean_df.WTI.pct_change().corr(clean_df.USD_Index.pct_change()),
 "corr(WTI,SP500)":clean_df.WTI.pct_change().corr(clean_df.SP500.pct_change()),
 "corr(WTI,XLE)":clean_df.WTI.pct_change().corr(clean_df.XLE.pct_change()),
})
summary.round(4)'''))

# ----------------------------------------------------------- plots -----------
cells.append(md("### 6 · Exploratory plots\n\n**(A) Distributional** — return distribution vs Normal + Q-Q."))
cells.append(code(
'''fig,ax=plt.subplots(1,2,figsize=(14,4.8))
sns.histplot(r,bins=80,stat="density",kde=True,ax=ax[0],color="#3b6ea5")
x=np.linspace(r.min(),r.max(),400)
ax[0].plot(x,stats.norm.pdf(x,r.mean(),r.std()),"r--",lw=2,label="Normal fit")
ax[0].set(title="WTI weekly log-returns vs Normal",xlabel="log-return"); ax[0].legend()
stats.probplot(r,dist="norm",plot=ax[1]); ax[1].set_title("Normal Q-Q (fat tails)")
plt.tight_layout(); plt.show()'''))

cells.append(md("**(B) Time series** — price with events, returns, rolling volatility."))
cells.append(code(
'''fig,ax=plt.subplots(3,1,figsize=(14,10),sharex=True)
ax[0].plot(clean_df.index,clean_df[TARGET],color="#1a1a1a",lw=1)
ax[0].set(title="WTI spot with events",ylabel="USD/bbl")
for d,l in [("2008-09-15","GFC"),("2014-11-27","glut"),("2020-04-20","COVID"),("2022-02-24","Ukraine")]:
    ax[0].axvline(pd.Timestamp(d),color="crimson",ls=":",lw=1.1)
    ax[0].text(pd.Timestamp(d),clean_df[TARGET].max()*.9,l,rotation=90,color="crimson",va="top",fontsize=9)
ax[1].plot(r.index,r,color="#3b6ea5",lw=.6); ax[1].set(title="Weekly log-returns (clustering)",ylabel="ret")
rv=r.rolling(13).std()*np.sqrt(52)
ax[2].plot(rv.index,rv,color="#b5651d",lw=1.2); ax[2].set(title="13-week rolling annualised vol",ylabel="vol")
plt.tight_layout(); plt.show()'''))

cells.append(md("**(C) Multivariate** — correlation heatmap, oil-vs-USD, rolling correlation."))
cells.append(code(
'''rets=clean_df[["WTI","Brent","USD_Index","SP500","Gold","XLE","NatGas"]].pct_change().dropna()
lvl =clean_df[["UST10Y","VIX","OVX","US_Production","US_Inventory","GPR"]].pct_change().add_suffix("_chg")
corr=pd.concat([rets,lvl],axis=1,sort=False).dropna().corr()
plt.figure(figsize=(11,9))
sns.heatmap(corr,annot=True,fmt=".2f",cmap="coolwarm",center=0,square=True,annot_kws={"size":7})
plt.title("Correlation of weekly changes"); plt.tight_layout(); plt.show()

fig,ax=plt.subplots(1,2,figsize=(14,4.8))
ax[0].scatter(clean_df.USD_Index.pct_change(),clean_df.WTI.pct_change(),s=7,alpha=.4,color="#3b6ea5")
ax[0].set(title="WTI vs USD weekly change",xlabel="USD chg",ylabel="WTI chg")
roll=clean_df.WTI.pct_change().rolling(52).corr(clean_df.USD_Index.pct_change())
ax[1].plot(roll.index,roll,color="#b5651d"); ax[1].axhline(0,color="k",lw=.8)
ax[1].set(title="52-week rolling corr(WTI,USD)",ylabel="corr")
plt.tight_layout(); plt.show()'''))

cells.append(md("**(D) Autocorrelation** — returns (≈ white noise) vs squared returns (volatility memory)."))
cells.append(code(
'''lags=26; ac_r=acf(r,nlags=lags,fft=True)[1:]; ac_r2=acf(r**2,nlags=lags,fft=True)[1:]
ci=1.96/np.sqrt(len(r))
fig,ax=plt.subplots(1,2,figsize=(14,4.6),sharey=True)
ax[0].bar(range(1,lags+1),ac_r,color="#3b6ea5"); ax[0].axhline(ci,color="r",ls="--"); ax[0].axhline(-ci,color="r",ls="--")
ax[0].set(title="ACF of returns",xlabel="lag")
ax[1].bar(range(1,lags+1),ac_r2,color="#b5651d"); ax[1].axhline(ci,color="r",ls="--"); ax[1].axhline(-ci,color="r",ls="--")
ax[1].set(title="ACF of squared returns",xlabel="lag")
plt.tight_layout(); plt.show()'''))

# --------------------------------------------------- Step 8 answers ----------
cells.append(md(
"""### 7 · Answers to the Step 8 questions

The numbers quoted below are read directly from the `summary` table above (they
refresh automatically when the notebook is re-run on the live data)."""))
cells.append(code(
'''ek   = summary["excess kurtosis"]
sk   = summary["skewness"]
c_usd= summary["corr(WTI,USD)"]
print("(a) DIFFERENT FROM OTHER ASSETS")
print("    - Sharp geopolitical/supply SPIKES (see event lines in plot B1).")
print("    - Strong VOLATILITY CLUSTERING: squared-return Ljung-Box p =",
      f"{summary['Ljung-Box p, squared returns']:.1e} (reject 'no memory').")
print("    - Seasonality (heating/driving demand) + storage/convenience-yield effects,")
print("      incl. the Apr-2020 negative-price episode driven by storage limits.")
print()
print(f"(b) DISTRIBUTION OF RETURNS: leptokurtic & fat-tailed. Excess kurtosis = {ek:.2f}")
print(f"    (>0 vs Normal), skewness = {sk:.2f}; Jarque-Bera p = "
      f"{summary['Jarque-Bera p']:.1e} -> Normality strongly rejected.")
print("    Better described by a Student-t / mixture with jumps.")
print()
print("(c) AUTOCORRELATION: raw returns show little linear autocorrelation")
print(f"    (near white noise), but SQUARED returns are highly autocorrelated")
print("    (ARCH/volatility persistence) - see plot D.")
print()
print("(d) OTHER STYLIZED FACTS")
print(f"    - Prices non-stationary in levels (ADF p = {summary['ADF level p (price)']:.2f}),")
print(f"      stationary in returns (ADF p = {summary['ADF p (returns)']:.1e}).")
print(f"    - Inverse USD link: corr(WTI,USD) = {c_usd:.2f}; pro-cyclical with equities/energy;")
print("      inventories & production act as fundamental anchors; term structure flips")
print("      between contango and backwardation.")'''))

cells.append(md(
"""These findings motivate the modelling choice in GWP2: a **Bayesian network** can
encode the directed dependencies sketched below (geopolitical risk, the dollar,
inventories and industrial demand feeding the oil price, which in turn drives
oil-volatility and the energy sector), while remaining robust to the non-Normal,
regime-switching behaviour that defeats a single linear model."""))
cells.append(code(
'''import networkx as nx
G=nx.DiGraph(); G.add_edges_from([
 ("GPR","WTI"),("USD_Index","WTI"),("US_Inventory","WTI"),
 ("US_Production","US_Inventory"),("IndProd","WTI"),("WTI","OVX"),
 ("WTI","XLE"),("WTI","CPI"),("UST10Y","USD_Index"),("SP500","WTI"),("VIX","SP500")])
pos=nx.spring_layout(G,seed=7,k=1.1)
plt.figure(figsize=(10,7))
nx.draw_networkx(G,pos,node_color="#cfe0f2",node_size=2000,font_size=9,
                 edge_color="#666",arrows=True,arrowsize=16,width=1.3)
plt.title("Candidate causal skeleton for the oil-price Bayesian network (GWP2 preview)")
plt.axis("off"); plt.tight_layout(); plt.show()'''))

cells.append(md(
"""---
*Reproducibility:* synthetic seed `20240628`; weekly Friday grid, 2005–2024.
Set `USE_LIVE = True` and re-run top-to-bottom to reproduce every figure on the
real FRED/Yahoo data for the final submission."""))

nb = new_notebook(cells=cells)
nb.metadata = {"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},
               "language_info":{"name":"python"}}

print("executing notebook ...")
ep = ExecutePreprocessor(timeout=600, kernel_name="python3")
ep.preprocess(nb, {"metadata": {"path": "."}})
with open("GWP1_Oil_BayesianNetwork.ipynb", "w") as f:
    nbf.write(nb, f)
print("wrote GWP1_Oil_BayesianNetwork.ipynb  (", len(nb.cells), "cells )")
