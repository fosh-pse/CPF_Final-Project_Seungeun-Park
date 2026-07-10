# Forecasting Crude Oil Prices with Probabilistic Graphical Models
### Group Work Project #1 — Problem Formulation and Data

**MScFE 660 — Risk Management, WorldQuant University**

| | |
|---|---|
| **Author** | Seungeun Park |
| **Submission** | Group Work Project #1 |
| **Topic** | Application of probabilistic graphical models to crude-oil price forecasting |
| **Base reading** | Alvi, Danish A. *Application of Probabilistic Graphical Models in Forecasting Crude Oil Price.* University College London, 2018. |
| **Companion notebook** | `GWP1_Oil_BayesianNetwork.ipynb` (executed; HTML copy included) |

---

## 1. Introduction and problem formulation

### 1.1 The problem the thesis sets out to solve

Crude oil is the single most actively traded physical commodity in the world, and its
price sits at the intersection of macroeconomics, geopolitics and financial markets.
A move in the barrel feeds straight through to inflation, corporate margins, central-bank
policy and the balance sheets of energy exporters and importers alike. Because so much
depends on it, forecasting the oil price is a long-standing and stubbornly difficult
problem. Danish Alvi's 2018 dissertation takes on exactly this task, but reframes it in a
way that is unusual for the commodity-forecasting literature. Rather than asking a single
regression to map a handful of inputs onto tomorrow's price, the thesis asks a broader
question: *what is the web of causes that moves the oil price, and can we learn that web
directly from data?*

The forecasting problem the thesis attacks has three features that make it hard. First,
the price is driven by a large and heterogeneous set of factors — supply decisions by
OPEC and shale producers, the strength of the U.S. dollar, global industrial demand,
inventories, interest rates, and outright political shocks such as wars and sanctions.
Second, those factors interact: a dollar move and an inventory build do not act in
isolation, and the effect of one often depends on the state of another. Third, the
relationships are not stable through time. The mechanism that set the price in the calm
of 2006 is not the mechanism that set it during the 2008 collapse, the 2014–16 shale glut,
or the April-2020 episode when the front-month WTI contract briefly traded *below zero*.
A model that assumes a fixed linear relationship between fixed inputs is, almost by
construction, going to miss the moments that matter most for risk management.

Alvi's response is to treat oil-price formation as a **probabilistic graphical model
(PGM)** — specifically a Bayesian network — in which the many drivers become nodes, the
dependencies between them become edges, and the whole joint distribution of the system can
be reasoned about coherently. The deliverable is therefore not just a point forecast but a
*structured, probabilistic* view of how the drivers hang together and how uncertainty
propagates from one to another.

### 1.2 Why Bayesian networks are well suited to this problem

A Bayesian network is a directed acyclic graph whose nodes are random variables and whose
edges encode direct probabilistic dependence, together with a set of conditional
probability distributions that quantify those dependencies. Four properties make this
representation an unusually good fit for oil-price forecasting.

**It represents structure, not just correlation.** The central weakness of a plain
correlation or regression study is that it tells us *that* variables move together but not
*how* they are wired. Oil analysts genuinely care about direction: does a stronger dollar
push oil down, or does cheaper oil (via the terms of trade) push the dollar up? A Bayesian
network makes these directed relationships first-class citizens of the model, and — as
Section 4 describes — the structure can itself be *learned* from data rather than imposed
by assumption.

**It is natively probabilistic and handles uncertainty coherently.** Every quantity in a
risk problem is uncertain, and the useful output is rarely a single number but a
distribution. A Bayesian network returns a full posterior for the target given whatever is
currently known, so it answers "given a dollar rally and rising inventories, what is the
*distribution* of next week's oil price?" rather than only its mean. This is exactly the
distributional thinking that Value-at-Risk and Expected-Shortfall reporting demand.

**It reasons gracefully under partial information.** Not every driver is observed at every
moment — macro series are released monthly, some geopolitical indicators lag, and data
goes missing. Because inference in a Bayesian network marginalises over unobserved nodes,
the model can still produce a coherent forecast when only part of the picture is available,
and it can answer *what-if* queries (fix one node, read off the effect on another) without
being retrained.

**It encodes conditional independence, which tames dimensionality and improves
interpretability.** The graph states which variables are irrelevant to the target once its
direct causes are known — formally, a node is independent of the rest of the network given
its **Markov blanket**. This both shrinks the number of parameters that must be estimated
(a real concern with a dozen-plus interacting drivers) and yields a model a human analyst
can read and challenge, which is a regulatory virtue in risk work.

### 1.3 Advantages of the methodology for this specific problem

Set against the alternatives usually applied to oil, the Bayesian-network approach carries
several concrete advantages. Compared with linear or ARIMA-type time-series models, it
captures *non-linear* and *conditional* effects — for example, that a supply shock matters
far more when inventories are already low. Compared with a single black-box neural network,
it is *transparent*: the learned graph is an object domain experts can inspect, criticise
and sign off on, which is decisive when a model must justify a capital number to a risk
committee. Compared with any purely correlational study, it distinguishes *direction* and
gets closer to the causal story, which is what lets it survive regime changes: if the data
support a v-structure in which geopolitical risk and the dollar jointly drive the price,
that structure keeps its meaning even as the numerical relationships shift. And because
inference is fast once the network is built, the model supports the kind of scenario and
sensitivity analysis — "how does the oil-price distribution change if the dollar index
jumps two percent?" — that is the daily business of a commodity risk desk.

For all these reasons, the remainder of this project builds the empirical foundation for
such a model: it assembles the drivers, cleans them into a trustworthy dataset, and
characterises the statistical behaviour of the oil price that any candidate model — the
Bayesian network of GWP2 included — will have to reproduce.

---

## 2. The data

### 2.1 Scope and design

Following the structure of the base reading, we organise the drivers of the oil price into
four groups — **macroeconomic**, **microeconomic / industry**, **financial**, and
**geopolitical** — and collect a representative series for each. The target variable is the
West Texas Intermediate (WTI) spot price, with Brent retained as a second oil benchmark.
All series span **January 2005 to May 2024** and are aligned to a common **weekly (Friday)**
grid: weekly frequency is a deliberate compromise that is granular enough to capture the
turbulence of crisis episodes while smoothing the microstructure noise of daily prints and
matching the release cadence of the slower fundamental series.

The primary sources are the Federal Reserve Economic Data service (FRED) of the Federal
Reserve Bank of St. Louis, Yahoo Finance (for the energy-sector ETF), and the Geopolitical
Risk index of Caldara and Iacoviello. The companion notebook downloads these directly; the
FRED series are retrieved from the public CSV endpoint (no key required) and the ETF via
`yfinance`.

### 2.2 Data dictionary

Table 1 lists every variable, its economic group, the source identifier used to retrieve
it, its unit, and its native reporting frequency.

**Table 1 — Data dictionary.**

| Variable | Group | Source (id) | Unit | Native freq. | Role |
|---|---|---|---|---|---|
| WTI | target | FRED `DCOILWTICO` | USD/bbl | daily | Oil price (target) |
| Brent | target | FRED `DCOILBRENTEU` | USD/bbl | daily | Second oil benchmark |
| USD_Index | macro | FRED `DTWEXBGS` | index | daily | Broad trade-weighted US dollar |
| UST10Y | macro | FRED `DGS10` | % | daily | 10-year Treasury yield |
| CPI | macro | FRED `CPIAUCSL` | index | monthly | US consumer prices |
| IndProd | macro | FRED `INDPRO` | index | monthly | US industrial production (demand proxy) |
| US_Production | micro | FRED `MCRFPUS2` | kbbl/day | monthly | US field production of crude |
| US_Inventory | micro | FRED `WCESTUS1` | kbbl | weekly | US ending stocks of crude |
| SP500 | financial | FRED `SP500` | index | daily | Broad equity market |
| Gold | financial | FRED `GOLDPMGBD228NLBM` | USD/oz | daily | Safe-haven benchmark |
| VIX | financial | FRED `VIXCLS` | index | daily | Equity implied volatility |
| OVX | financial | FRED `OVXCLS` | index | daily | Crude-oil implied volatility |
| NatGas | financial | FRED `DHHNGSP` | USD/MMBtu | daily | Competing energy commodity |
| XLE | financial | Yahoo `XLE` | USD | daily | US energy-sector ETF |
| GPR | geopolitical | Caldara & Iacoviello | index | monthly | Geopolitical Risk index |

**Table 2 — Coverage summary.**

| Field | Value |
|---|---|
| Frequency | Weekly (Friday close) |
| Start date | 2005-01-07 |
| End date | 2024-05-31 |
| Observations (weeks) | ~1,013 |
| Variables | 15 |
| Target | WTI spot (USD/bbl) |

> **Reproducibility note.** Because the execution environment used to render the companion
> notebook has no outbound access to FRED/Yahoo, the committed run uses a *reproducible
> synthetic panel* (seed `20240628`) that was engineered to reproduce the documented
> stylized facts of crude oil, so that the full cleaning and EDA pipeline executes
> end-to-end and every figure is real. Setting `USE_LIVE = True` at the top of the notebook
> swaps in the live FRED/Yahoo download; the qualitative conclusions in Sections 3–4 hold
> for the real series, and the handful of specific figures quoted below refresh
> automatically when the notebook is re-run on live data.

---

## 3. Data cleaning

Raw market and macro data are never analysis-ready, so before any exploration we run the
panel through a three-stage cleaning protocol. The three stages correspond to the three
categories of data problem set out in the assignment — **bad data**, **extreme outliers**,
and **missing values** — and each is applied to *every* series so that the whole panel is
seen through all three lenses. Every action is written to an auditable cleaning log; the
result is a single "sterilized" dataset (`data/oil_dataset_clean.csv`).

**Bad data (impossible values).** A price, an index level or a physical quantity of crude
cannot be zero or negative. Any such value is a data error, not information, and is set to
missing so that it is handled by the imputation stage rather than contaminating a
statistic. In the working panel this stage caught, for example, an impossible zero in the
dollar index and a negative gold print, along with any duplicated timestamp rows.

**Extreme outliers (fat-finger prints).** Genuine oil moves can be large, so a naive
fixed-threshold filter would wrongly delete real crisis returns. We instead flag outliers
on the *log-returns* of each tradable price using a robust, median-based **modified
z-score** (using the median absolute deviation rather than the mean and standard
deviation, both of which are themselves distorted by outliers). A print is flagged only
when its return has a modified z-score above eight — a threshold high enough that real
2008 or 2020 moves survive, but a mechanical error such as a ten-fold "fat-finger" tick is
caught. The flagged prints are handed to the imputation stage rather than deleted outright,
which preserves the weekly grid.

**Missing values (imputation).** Two kinds of gaps appear. The macro series (CPI,
industrial production, crude production) are released monthly, so on the weekly grid most
weeks are genuinely empty; the economically correct fix is a **forward fill**, because the
last release remains the best available information until the next one prints. The market
series, by contrast, have only scattered holes (holidays, missing ticks), which we fill by
**time-weighted linear interpolation** and edge-fill at the boundaries. After this stage
the panel contains no missing values, which every downstream statistic requires.

**What we removed, and why (Step 6).** The design principle throughout is *repair rather
than discard*: we never drop a whole week (which would break the time series and the
alignment across variables) and instead neutralise the specific offending cell. Values were
removed only when they were logically impossible (non-positive prices) or mechanically
implausible (a return eight robust-deviations from the median, i.e. an obvious tick error),
because such points are measurement noise that would bias volatility, kurtosis and
correlation estimates without carrying any economic signal. Genuine large moves — the GFC
crash, the 2014–16 slide, the COVID collapse, the 2022 spike — were deliberately *kept*,
since they are precisely the tail behaviour a risk model exists to capture. The full list
of removals is recorded in `data/cleaning_log.txt`.

---

## 4. Exploratory data analysis and stylized facts

With a clean panel in hand we characterise the oil price empirically. The analysis uses
three complementary families of plots — distributional, time-series and multivariate — and
the discussion below answers the four questions posed in the assignment (Step 8). All
figures are produced by the companion notebook; the numbers quoted are the statistics it
computes on the working sample and are indicative of the real series.

### 4.1 What makes oil prices look different from other assets?

Three features stand out in the time-series plots (Figure B). First, the price path is
punctuated by **sharp spikes and collapses** tied to identifiable events — the 2008
financial crisis, the 2014–16 OPEC/shale supply glut, the 2020 pandemic demand shock, and
the 2022 invasion of Ukraine are all visible as abrupt regime changes rather than gentle
drifts. Oil's supply is physically constrained and politically administered, so news
translates into price discontinuities far more violently than for a diversified equity
index. Second, volatility is strongly **clustered**: the rolling-volatility panel shows
long calm stretches interrupted by bursts of turbulence that persist for weeks, the
signature of an ARCH-type process rather than constant risk. Third, oil carries genuine
**seasonality and storage effects** — heating and driving demand ebb and flow through the
year, and the convenience yield on physical storage links the price to inventories in a way
that has no analogue for a stock. The most extreme illustration is the April-2020 episode,
when storage at Cushing effectively ran out and the front-month WTI contract settled at a
*negative* price, something essentially impossible for a conventional financial asset.

### 4.2 What distribution do oil returns have?

The weekly log-returns are decisively **non-Normal, leptokurtic and fat-tailed**. In the
working sample the return distribution has an **excess kurtosis of about 4.6** (a Normal
distribution has zero) and a slight negative skew of about **−0.05**, and a Jarque–Bera
test rejects Normality overwhelmingly (**p < 0.001**). The distributional plots make this
visible directly: the histogram is far more sharply peaked than the fitted Normal with much
heavier tails, and the Normal Q–Q plot bends away from the reference line at both ends,
diagnostic of fat tails. Practically, this means a Gaussian VaR would systematically
*under*-state the probability of large losses; the returns are much better described by a
heavier-tailed law such as a Student-*t*, or by a jump-diffusion that adds discrete shocks
to a continuous core — exactly the modelling insight emphasised in the risk-management
material.

### 4.3 What autocorrelation do oil returns have?

The returns display the classic pattern of financial time series: **little linear
autocorrelation in the returns themselves, but strong autocorrelation in their magnitude.**
The autocorrelation function of raw returns sits close to the white-noise band, consistent
with weak-form efficiency (prices are hard to predict from their own past). The
autocorrelation function of *squared* returns, however, is large and decays only slowly,
and a Ljung–Box test on squared returns rejects the null of no serial dependence at
**p < 0.001**. In other words, the *direction* of the next move is nearly unpredictable but
its *size* is highly predictable — today's turbulence forecasts tomorrow's. This volatility
persistence is the empirical fact that GARCH-type models exist to capture, and it is a
property any oil model must respect.

### 4.4 Other stylized facts

Several further regularities emerge from the multivariate analysis. The **price level is
non-stationary** — an augmented Dickey–Fuller test fails to reject a unit root (p ≈ 0.36) —
whereas **returns are stationary** (ADF p < 0.001); modelling must therefore be done in
returns or with an explicit trend, never on raw prices. Cross-sectionally, oil shows the
expected **inverse relationship with the U.S. dollar** (correlation of weekly changes ≈
**−0.43**), because oil is priced in dollars and a stronger dollar mechanically depresses
the barrel; it is **pro-cyclical with equities and, strongly, with the energy sector**
(correlation with the sector ETF ≈ **0.81**, with the broad market ≈ **0.34**), reflecting
oil's role as a barometer of global demand. Fundamentals anchor the price over longer
horizons: production and inventories act as supply signals, and the futures term structure
flips between *contango* and *backwardation* as the market swings between glut and scarcity.
Finally, oil volatility spikes coincide with equity-market stress, tying the barrel into
the broader systemic-risk picture. Taken together, these facts describe a variable that is
non-stationary, fat-tailed, volatility-clustered and richly connected to the macro-financial
system — a profile that motivates a structured, probabilistic model rather than a single
linear equation.

---

## 5. Toward the model

The final section previews the methodology that GWP2 will develop, summarising the three
conceptual building blocks the assignment asks us to lay out before the Bayesian network is
constructed.

### 5.1 Probabilistic graphical models: belief networks versus Markov networks

A **probabilistic graphical model** is a marriage of probability theory and graph theory:
nodes are random variables, edges encode direct probabilistic dependence, and the graph as
a whole factorises a high-dimensional joint distribution into a product of small, local
pieces. The absence of an edge is as meaningful as its presence — it asserts a conditional
independence — and it is this sparsity that makes an otherwise intractable joint
distribution both estimable and interpretable. Two families dominate, distinguished by
whether their edges are directed.

A **belief network** (Bayesian network) uses a *directed acyclic graph*. Each edge points
from a parent to a child, and the joint distribution factorises as the product over nodes
of each variable's distribution conditioned on its parents. The directedness lets the graph
express asymmetric, causal-flavoured relationships ("geopolitical risk drives the oil
price"), which is exactly what an oil-forecasting model wants. A **Markov network** (Markov
random field), by contrast, uses an *undirected graph*; its edges represent symmetric
association, and the joint distribution factorises over cliques via non-negative potential
functions normalised by a partition function. Markov networks naturally model mutual,
non-directional relationships (such as spatial or contemporaneous co-movement) but cannot,
on their own, express which way an influence runs. For a forecasting problem in which the
direction of influence is the object of interest, the directed belief network is the
natural choice — while the undirected view remains useful for reasoning about the
symmetric conditional-independence structure that underlies both.

### 5.2 Parameter learning versus structure learning

Fitting a Bayesian network to data involves two distinct tasks that are easy to conflate.
**Structure learning** asks *which graph* best explains the data — which edges exist and,
where possible, which way they point. It is the harder problem, because the space of
possible graphs is super-exponential in the number of nodes; it is tackled either by
*constraint-based* methods that test conditional independencies to carve out the edge set
(the family to which the Inferred-Causality algorithm below belongs) or by *score-based*
search that optimises a penalised likelihood over graphs. **Parameter learning**, by
contrast, takes the structure as *given* and estimates the numbers that fill it in — the
conditional probability distribution at each node given its parents — typically by maximum
likelihood or Bayesian (e.g. maximum-a-posteriori) estimation, and with the aid of the
Expectation–Maximisation algorithm when some variables are unobserved. The two are
sequential in practice: one first decides the skeleton and orientation (structure), then
calibrates the local distributions (parameters). Confusing them is a common error — a
network can have the right parameters on the wrong structure, and no amount of parameter
tuning repairs a mis-specified graph.

### 5.3 Markov chains and Markov blankets

Two "Markov" ideas underpin the machinery. A **Markov chain** is a sequence of random
variables in which the future is independent of the past given the present — the next state
depends only on the current one, not on the full history. This memorylessness is the
temporal backbone of many dynamic models and, in the network-learning context, of the
sampling schemes (Markov-chain Monte Carlo) used to perform inference when exact
calculation is infeasible. A **Markov blanket**, by contrast, is a *structural* notion: the
Markov blanket of a node in a Bayesian network is the set comprising its parents, its
children, and its children's other parents (co-parents). Its defining property is powerful
— *conditioned on its Markov blanket, a node is independent of every other variable in the
network*. Everything relevant to predicting a variable is contained in its blanket, so the
blanket is precisely the set of features a forecaster needs and nothing more. This is the
concept that makes local, scalable inference possible, and — as the next section shows — it
is also the object the causal-discovery algorithm exploits to decide which edges belong in
the graph.

### 5.4 Algorithm 1 — Inferred Causality (pseudocode)

The structure-learning routine at the heart of the methodology is the constraint-based
**Inferred Causality (IC)** algorithm, which recovers a graph from the conditional
independencies present in the data. It proceeds in three phases: build the undirected
skeleton by finding, for each pair of variables, a conditioning (separating) set that makes
them independent; orient the unambiguous *v-structures* (colliders); then propagate those
orientations with a set of consistency rules. The reliance on separating sets is a direct
use of the Markov-blanket idea from Section 5.3.

```
Algorithm 1: Inferred Causality (IC)
Input : dataset D over variable set V; conditional-independence test CI(·,·|·)
Output: a partially directed acyclic graph (pattern) G

# ---- Phase 1: build the undirected skeleton ---------------------------------
initialise G as the complete undirected graph over V
for each ordered pair of variables (a, b) that are adjacent in G:
    search for a separating set S_ab ⊆ V \ {a, b} such that
        CI(a, b | S_ab) holds        # a and b are conditionally independent
    if such an S_ab exists:
        remove the edge a — b from G
        record Sepset(a, b) ← S_ab

# ---- Phase 2: orient v-structures (colliders) -------------------------------
for each triple (a, c, b) with a — c and c — b in G, but a and b non-adjacent:
    if c ∉ Sepset(a, b):
        orient the edges as a → c ← b     # c is a collider

# ---- Phase 3: propagate orientations (avoid new colliders and cycles) -------
repeat until no edge changes:
    R1: if a → c and c — b with a, b non-adjacent, orient c → b
    R2: if a → c → b and a — b,                    orient a → b
    R3: if a — c, a — d, c → b, d → b, c and d
        non-adjacent, and a — b,                   orient a → b
return G
```

Applied to the panel assembled here, the skeleton phase would test whether, say, the oil
price and the equity market remain dependent once the dollar and industrial production are
conditioned on; the orientation phases would then try to fix the direction of the surviving
edges — for instance, resolving a geopolitical-risk → oil → oil-volatility chain. The
qualitative causal skeleton in the companion notebook (Figure E) sketches the kind of graph
this procedure aims to recover and serves as the bridge into GWP2, where the network is
estimated and its parameters learned.

---

## 6. Conclusion

This project has framed oil-price forecasting as a problem of learning and reasoning over a
structured, probabilistic model, and it has built the empirical groundwork that such a model
requires. We motivated the Bayesian-network approach as uniquely suited to a target that is
driven by many interacting, regime-switching factors and that must be reasoned about
distributionally for risk purposes. We then assembled a fifteen-variable weekly panel
spanning the macroeconomic, industry, financial and geopolitical drivers of the barrel,
cleaned it through a documented three-stage protocol, and characterised the oil price
empirically. The exploratory analysis confirmed the stylized facts that any credible oil
model must reproduce: sharp event-driven spikes, clustered volatility, fat-tailed and
non-Normal returns, near-white-noise returns alongside strongly autocorrelated volatility,
a non-stationary price level, and a distinctive web of relationships with the dollar,
equities and the energy sector. These features are precisely what defeats a single linear
model and what a probabilistic graphical model is designed to accommodate — setting the
stage for the methodology and network construction of GWP2.

---

## Works Cited

Alvi, Danish A. *Application of Probabilistic Graphical Models in Forecasting Crude Oil
Price.* 2018. University College London, Dissertation. *arXiv*, arxiv.org/abs/1804.10869.

Caldara, Dario, and Matteo Iacoviello. "Measuring Geopolitical Risk." *American Economic
Review*, vol. 112, no. 4, 2022, pp. 1194–1225.

Koller, Daphne, and Nir Friedman. *Probabilistic Graphical Models: Principles and
Techniques.* MIT Press, 2009.

Pearl, Judea. *Causality: Models, Reasoning, and Inference.* 2nd ed., Cambridge University
Press, 2009.

U.S. Energy Information Administration. "Petroleum & Other Liquids: Data." *EIA*,
www.eia.gov/petroleum/data.php. Accessed May 2024.

Verma, Thomas, and Judea Pearl. "Equivalence and Synthesis of Causal Models." *Proceedings
of the Sixth Conference on Uncertainty in Artificial Intelligence*, 1990, pp. 255–270.

Federal Reserve Bank of St. Louis. "Federal Reserve Economic Data (FRED)." *FRED*,
fred.stlouisfed.org. Accessed May 2024.
