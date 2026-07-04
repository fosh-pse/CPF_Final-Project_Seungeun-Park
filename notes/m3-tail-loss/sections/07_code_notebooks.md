## 코드 완전 해설 ② — DeepVaR 노트북

앞에서 배운 **DeepVaR**의 이론을, 이제 실제 코드로 한 줄씩 뜯어볼 차례예요. 이 노트북의 큰 흐름은 딱 하나로 요약됩니다. **GluonTS**라는 라이브러리로 **확률예측(probabilistic forecasting)**을 하고, 그 예측 분포에서 **분위수(quantile)**를 뽑아 **VaR(Value at Risk)**로 바꾸는 파이프라인이에요. 즉 "미래 수익률이 이런 분포로 나올 것 같다"를 신경망이 먼저 그려 주면, 우리는 그 분포의 꼬리를 잘라 위험을 읽어 냅니다.

> 💡 **쉬운 설명** — 보통의 예측 모델은 "내일 수익률은 $-0.3\%$"처럼 숫자 하나만 내놓아요. 하지만 DeepVaR는 "내일 수익률의 분포는 이렇게 생겼다"를 통째로 내놓습니다. 분포가 있으면 그 아래쪽 $1\%$ 지점을 잘라 "$99\%$ 확신하는 최악의 손실"을 바로 읽을 수 있죠. 그게 VaR입니다.

### 데이터 준비(셀 0~7)

먼저 필요한 도구들을 불러옵니다.

```python
from gluonts.dataset.common import ListDataset
from gluonts.model import deepar
from gluonts.mx.trainer import Trainer
from gluonts.evaluation import Evaluator
```

한 줄씩 볼게요. `ListDataset`은 GluonTS가 요구하는 **전용 데이터 형식**이에요. 우리가 가진 판다스 표를 이 그릇에 담아야 모델이 먹을 수 있습니다. `deepar`는 이 노트북의 심장인 **DeepAR** 모델 모듈, `Trainer`는 학습을 돌리는 엔진(에폭 수·학습률 같은 걸 쥐고 있음), `Evaluator`는 예측이 얼마나 맞았는지 채점하는 평가기예요.

```python
paths = ["data/AUDUSD1440.csv","data/GBPUSD1440.csv",
         "data/USDJPY1440.csv","data/EURUSD1440.csv"]
```

논문이 다룬 **4개 통화쌍(currency pair)**의 데이터 경로예요. 호주달러/미국달러(AUDUSD), 영국파운드/미국달러(GBPUSD), 미국달러/일본엔(USDJPY), 유로/미국달러(EURUSD)입니다. 파일명 뒤의 `1440`은 분 단위 캔들 주기(하루 = $1440$분), 즉 **일봉 데이터**라는 뜻이에요.

```python
def read_asset_data(path):
    asset = pd.read_csv(path, delimiter='\t', usecols=[0,4], names=['datetime','price'])
    asset.datetime = pd.to_datetime(asset.datetime)     # 문자열→날짜
    asset.set_index('datetime', inplace=True)           # 날짜를 인덱스로
    asset_name = "".join(re.findall("[a-zA-Z]+", path))[-9:-3]  # 파일명에서 통화쌍 이름
    asset.rename(columns={"price": asset_name}, inplace=True)
    return asset[[asset_name]]
```

이 함수는 CSV 하나를 깔끔한 표로 읽어 주는 도우미예요. 한 줄씩 따라가 볼게요.

- `pd.read_csv(..., delimiter='\t', usecols=[0,4], names=['datetime','price'])` — 탭(`\t`)으로 나뉜 파일에서 **0번 열(시간)**과 **4번 열(종가)**만 골라 읽고, 각각 `datetime`·`price`라는 이름을 붙입니다. 나머지 시가·고가·저가 열은 버려요.
- `pd.to_datetime(asset.datetime)` — 문자열로 된 날짜("2015.01.02" 같은 것)를 파이썬이 진짜 **날짜 자료형**으로 이해하도록 변환합니다.
- `set_index('datetime', inplace=True)` — 그 날짜를 표의 **인덱스(행 이름)**로 삼아요. 이러면 시계열로 다루기 편해집니다.
- `re.findall("[a-zA-Z]+", path)` — 경로 문자열에서 **알파벳만** 골라냅니다. 여기에 `[-9:-3]` 슬라이싱을 걸어 "AUDUSD" 같은 여섯 글자 통화쌍 이름만 뽑아내죠.
- `rename(...)` 후 `return asset[[asset_name]]` — `price`라는 밋밋한 열 이름을 실제 통화쌍 이름으로 바꿔서, 이 열 하나만 돌려줍니다.

> 📝 **예시** — `"data/EURUSD1440.csv"`가 들어오면, 알파벳만 모아 `"dataEURUSDcsv"`가 되고 여기서 `[-9:-3]`을 자르면 정확히 `"EURUSD"`가 남아요. 그래서 반환되는 표의 열 이름이 `EURUSD`가 됩니다.

이렇게 4개 파일을 읽어 옆으로 붙이면 `prices`라는, 날짜가 인덱스이고 4개 통화쌍이 열인 가격표가 완성돼요. 다음은 이 가격을 수익률로 바꾸는 단계입니다.

```python
returns = prices.pct_change().dropna()          # 일별 수익률
returns = returns.asfreq(freq='1D', fill_value=0.0)  # DeepAR는 '일정 빈도' 요구
returns = returns[returns.index >= '2015-01-01']     # 최근 6년만
```

- `prices.pct_change()` — 어제 대비 오늘의 **변화율**, 즉 일별 수익률을 계산합니다. 수식으로는 $r_t = \dfrac{P_t - P_{t-1}}{P_{t-1}}$이에요. 여기서 $P_t$는 $t$일의 종가, $P_{t-1}$은 그 전날 종가입니다. 맨 첫날은 비교할 전날이 없어 `NaN`이 되므로 `dropna()`로 지웁니다.
- `asfreq(freq='1D', fill_value=0.0)` — 여기가 핵심이에요. **DeepAR**는 시계열이 **일정한 시간 간격(regular frequency)**으로 놓여 있길 요구합니다. 그런데 외환 데이터는 주말·공휴일에 구멍이 나 있죠. 그래서 `'1D'`(하루 간격)로 격자를 강제하고, 거래가 없어 빈 날은 수익률 $0$으로 메꿉니다.
- `returns.index >= '2015-01-01'` — 논문 설정에 맞춰 **2015년 이후 최근 약 6년치**만 남깁니다.

> ⚠️ **주의** — 빈 날을 $0$으로 채우는 건 "그날은 가격이 그대로였다"는 가정이에요. 편의를 위한 처리지만, 휴장일이 많으면 변동성을 실제보다 살짝 낮게 보이게 할 수 있습니다. DeepAR의 "일정 간격" 요구를 맞추려는 타협이라는 점을 기억해 두세요.

### DeepARModel 클래스(셀 8)

이제 DeepAR 모델을 쓰기 편하게 감싼 **래퍼 클래스(wrapper class)**를 봅니다. 데이터 변환·학습·예측을 한 묶음으로 정리해 둔 상자예요.

```python
class DeepARModel:
    def __init__(self, freq='1D', context_length=15, prediction_length=10,
                 epochs=5, learning_rate=1e-4, n_layers=2., dropout=0.1):
        # 논문 설정과 일치: 15일 컨텍스트, lr=0.0001, 2층, 드롭아웃 10%
```

생성자 `__init__`은 모델의 **하이퍼파라미터(hyperparameter)**를 정합니다. 하나씩 볼게요.

- `context_length=15` — 예측할 때 **과거 며칠을 참고**할지. 여기선 $15$일치를 봅니다.
- `prediction_length=10` — 한 번에 **며칠 앞을 예측**할지. 기본은 $10$일이에요.
- `epochs=5` — 전체 데이터를 몇 번 반복 학습할지.
- `learning_rate=1e-4` — 학습 보폭, 즉 $0.0001$. 한 번에 얼마나 크게 가중치를 고칠지를 정합니다.
- `n_layers=2.` — 순환신경망을 **2층** 쌓습니다.
- `dropout=0.1` — 학습 중 뉴런의 $10\%$를 무작위로 끄는 **드롭아웃(dropout)** 비율. 과적합을 막는 장치예요.

주석에도 적혀 있듯 이 값들은 모두 **논문 설정과 일치**합니다.

```python
    def list_dataset(self, ts, train=True):
        custom_dataset = self.df_to_np(ts)   # (시간×자산)→(자산×시간) 전치
        start = pd.Timestamp(ts.index[0], freq=self.freq)
        if train == True:
            ds = ListDataset([{'target': x, 'start': start}
                              for x in custom_dataset[:, :-self.prediction_length]], freq=self.freq)
        else:
            ds = ListDataset([{'target': x, 'start': start}
                              for x in custom_dataset], freq=self.freq)
        return ds
```

이 메서드는 판다스 표를 GluonTS의 `ListDataset`으로 번역합니다. 차근차근 볼게요.

- `self.df_to_np(ts)` — 표를 넘파이 배열로 바꾸면서 **(시간 × 자산)** 모양을 **(자산 × 시간)**으로 **전치(transpose)**합니다. GluonTS는 "한 행 = 하나의 시계열"을 원하기 때문에, 각 통화쌍이 하나의 행이 되도록 눕히는 거예요.
- `start = pd.Timestamp(ts.index[0], ...)` — 시계열의 **시작 날짜**를 잡아 둡니다. GluonTS는 각 시계열에 "언제부터 시작인지"를 함께 알려 줘야 하거든요.
- `if train == True:` 분기가 중요합니다. **학습용**이면 `custom_dataset[:, :-self.prediction_length]`로 **마지막 `prediction_length`일을 잘라냅니다.** 잘라낸 그 뒷부분이 나중에 채점할 **정답(미래)**으로 남죠. 반대로 **테스트용**(`else`)이면 자르지 않고 **전체**를 넣습니다.
- 각 원소 `{'target': x, 'start': start}` — `x`가 한 자산의 수익률 시계열(`target`), `start`가 시작일이에요. 즉 **자산 하나 = 시계열 하나**로 취급합니다.

> 💡 **쉬운 설명** — 학습용에서 뒷부분을 잘라 두는 건 시험 공부에 비유할 수 있어요. 뒤 $10$일을 가려 놓고 앞부분만 보여 주며 "가려진 뒤를 맞혀 봐"라고 시키는 거죠. 그래야 모델이 진짜 미래를 예측하는 연습을 하게 됩니다.

```python
    def fit(self, ts):
        estimator = deepar.DeepAREstimator(
            prediction_length=self.prediction_length,
            context_length=self.context_length,
            freq=self.freq,
            trainer=Trainer(epochs=self.epochs, ctx="cpu",
                            learning_rate=self.learning_rate, num_batches_per_epoch=50),
            num_layers=self.n_layers,
            dropout_rate=self.dropout,
            cell_type='lstm',      # LSTM 셀
            num_cells=50           # 논문대로 층당 50셀(기본 40에서 상향)
        )
        list_ds = self.list_dataset(ts, train=True)
        predictor = estimator.train(list_ds)   # 학습 실행(내부에서 로그우도 최대화, Adam)
        return predictor
```

`fit`은 실제로 모델을 **학습**시키는 메서드예요.

- `deepar.DeepAREstimator(...)` — DeepAR 모델의 설계도를 만듭니다. 앞서 정한 예측 길이·컨텍스트·빈도가 그대로 들어가고, `Trainer(...)`로 학습 엔진을 붙여요. `ctx="cpu"`는 CPU로 돌리라는 뜻, `num_batches_per_epoch=50`은 한 에폭에 미니배치 $50$개를 처리하라는 지시입니다.
- `cell_type='lstm'` — 순환 셀로 **LSTM**을 씁니다. 앞 레슨에서 배운 그 장·단기 기억 셀이에요.
- `num_cells=50` — 각 층에 뉴런(셀)을 $50$개 둡니다. GluonTS 기본값 $40$에서 **논문대로 $50$으로 올린** 부분이에요.
- `estimator.train(list_ds)` — 학습을 실행합니다. 이 한 줄 안에서 DeepAR는 내부적으로 **로그우도(log-likelihood)를 최대화**하는 방향으로, **Adam** 최적화기를 써서 가중치를 조정해요.

> 🎯 **핵심** — `cell_type='lstm'`과 `num_cells=50`, 이 두 설정이 논문 재현의 핵심 포인트예요. DeepAR는 단순히 "다음 값"이 아니라 **다음 값의 확률분포(스튜던트 $t$ 분포)의 파라미터**를 예측하도록 학습됩니다. 그래서 로그우도를 최대화한다는 말은 "실제 관측이 나올 확률을 가장 높게 만드는 분포를 찾는다"는 뜻이에요.

```python
    def predict(self, ts):
        test_ds = self.list_dataset(ts, train=False)
        return self.estimator.predict(test_ds, num_samples=1000)   # 몬테카를로 샘플 1000개
```

`predict`는 학습된 모델로 미래를 예측합니다. 여기서 `num_samples=1000`이 결정적이에요. DeepAR는 예측을 **숫자 하나가 아니라 $1000$개의 표본**으로 뱉습니다. 이 $1000$개의 **몬테카를로(Monte Carlo) 샘플**을 히스토그램으로 그리면 그게 바로 미래 수익률의 **예측 분포**예요. 우리는 이 분포에서 분위수를 취해 VaR을 만들 겁니다.

> 📝 **예시** — 내일 EURUSD 수익률을 $1000$번 시뮬레이션했더니 대부분 $-1\%$에서 $+1\%$ 사이에 몰렸다고 해 봐요. 이 $1000$개를 작은 순서로 줄 세워 아래에서 $10$번째($1\%$ 지점) 값을 읽으면, 그게 $99\%$ 신뢰수준 VaR의 재료가 됩니다.

### var_p — 포트폴리오 VaR(셀 11)

이제 종목별 예측 분포를 **포트폴리오 전체의 VaR** 하나로 합치는 함수입니다.

```python
def var_p(predictions, returns, weights, days_ahead=0, alpha=95):
    V = np.zeros(len(weights))
    for i in range(len(weights)):
        if weights[i] < 0 :   # 숏 → 높은 수익률(alpha 퍼센타일)이 손실
            V[i] = weights[i] * np.percentile(predictions[i].samples[:,days_ahead], alpha)
        else:                 # 롱 → 낮은 수익률(100-alpha 퍼센타일)이 손실
            V[i] = weights[i] * np.percentile(predictions[i].samples[:,days_ahead], 100-alpha)
    R = returns.corr()        # 전체 기간 상관행렬(논문 125일 롤링과 어긋남)
    return -np.sqrt(V @ R @ V.T)   # 논문 식 15: -√(VRVᵀ)
```

핵심은 두 단계예요. **먼저 종목별 위험 벡터 $V$를 만들고, 그다음 상관행렬로 하나로 묶습니다.**

반복문 안에서 각 종목 $i$의 위험 $V_i$를 계산하는데, **포지션 방향에 따라 손실이 나는 꼬리가 반대**라는 점이 포인트입니다.

- `if weights[i] < 0` (**숏 포지션**) — 가격이 오르면 손해죠. 그래서 수익률 분포의 **위쪽 꼬리**, 즉 `np.percentile(..., alpha)`(예: $95$ 퍼센타일)의 높은 수익률이 손실 시나리오가 됩니다.
- `else` (**롱 포지션**) — 가격이 내리면 손해예요. 그래서 **아래쪽 꼬리** `np.percentile(..., 100-alpha)`(예: $5$ 퍼센타일)의 낮은 수익률이 손실이 됩니다.
- 어느 쪽이든 그 분위수에 비중 `weights[i]`를 곱해 $V_i$에 저장해요. `predictions[i].samples[:, days_ahead]`가 바로 앞서 만든 $1000$개 몬테카를로 샘플에서 `days_ahead`일째 값들을 꺼낸 것입니다.

그리고 마지막 두 줄에서 이들을 결합합니다. 수식으로는 논문 **식 15**예요.

$$
\text{VaR}_p = -\sqrt{\, V R V^{\top} \,}
$$

기호를 풀어 볼게요.

- $V$ — 방금 만든 **종목별 가중 VaR 벡터**. 각 성분이 $V_i$입니다.
- $V^{\top}$ — 그 벡터의 **전치(transpose)**.
- $R$ — 종목 간 **상관행렬(correlation matrix)**, 코드의 `returns.corr()`예요.
- $-\sqrt{\cdot}$ — 상관을 반영해 결합한 위험의 제곱근을 씌우고, 손실을 **음수 부호**로 표현합니다.

> ⚠️ **주의** — 여기 숨은 한계가 하나 있어요. `returns.corr()`는 **넘겨받은 전체 기간의 단일 상관행렬**을 씁니다. 그런데 논문은 원래 $125$일 **롤링(rolling) 상관**을 쓰도록 되어 있어요. 상관관계는 시장 국면에 따라 계속 변하는데, 전체 기간 하나의 값으로 고정하면 위기 때 급변하는 상관을 놓칠 수 있습니다.

> 🎯 **핵심** — 종목별 VaR을 그냥 더하면 안 되는 이유가 바로 이 식에 담겨 있어요. 종목들이 완벽히 같이 움직이지 않으므로($R$의 비대각 성분이 $1$보다 작음), 상관행렬을 끼운 $-\sqrt{V R V^{\top}}$가 단순 합보다 **작은(=현실적인)** 위험을 줍니다. 이게 **분산 효과(diversification effect)**예요.

### 벤치마크 VaRCalculation(셀 13)

DeepVaR가 잘하는지 비교하려면 **전통적 VaR 방법**들과 겨뤄 봐야겠죠. 이 클래스는 교과서에 나오는 세 가지 고전 기법을 담고 있어요.

```python
class VaRCalculation:
    def mc(self, x, alpha, n_sims=5000, seed=42):     # 몬테카를로 VaR
        np.random.seed(seed)
        sim_returns = np.random.normal(x.mean(), x.std(), n_sims)  # 정규분포 시뮬
        return np.sqrt(self.time*self.freq) * np.percentile(sim_returns, alpha)
    def vc(self, x, alpha):                            # 분산-공분산 VaR
        c = alpha / 100
        return np.sqrt(self.time*self.freq) * (x.std() * stats.norm.ppf(c))  # σ×z
    def hs(self, x, alpha):                            # 역사적 시뮬레이션 VaR
        return np.sqrt(self.time*self.freq) * np.percentile(x, 100 - alpha)  # 경험 분위수
```

세 메서드를 하나씩 볼게요.

- **`mc` — 몬테카를로(Monte Carlo) VaR.** 실제 데이터의 평균 `x.mean()`과 표준편차 `x.std()`를 가진 **정규분포에서 $5000$개를 뽑아**(시뮬레이션) 그중 분위수를 취합니다. `np.random.seed(seed)`는 매번 같은 난수가 나오게 고정해 **재현성**을 보장해요.
- **`vc` — 분산-공분산(variance-covariance) VaR.** 시뮬레이션 없이 공식 하나로 끝냅니다. `c = alpha/100`로 신뢰수준을 확률로 바꾸고, 표준편차에 **정규분포 분위수** `stats.norm.ppf(c)`를 곱해요. 이게 논문 **식 1**이자, 딱 $\sigma \times z$ 꼴입니다.
- **`hs` — 역사적 시뮬레이션(historical simulation) VaR.** 분포를 가정하지 않고 **실제 수익률 데이터**를 그대로 줄 세워 `np.percentile(x, 100-alpha)`, 즉 **경험 분위수(empirical quantile)**를 읽습니다. 가장 **비모수적(nonparametric)**인 방법이에요.

`vc`의 공식을 수식으로 쓰면 이렇습니다.

$$
\text{VaR}_{vc} = \sqrt{t \cdot f}\;\cdot\;\sigma \cdot \Phi^{-1}(c)
$$

기호를 풀면, $\sigma$는 수익률의 **표준편차**(변동성), $\Phi^{-1}(c)$는 신뢰수준 $c$에 해당하는 **표준정규분포의 분위수**(코드의 `stats.norm.ppf(c)`, 흔히 $z$-값), $t \cdot f$는 시간 관련 배수예요.

> 💡 **쉬운 설명** — 세 방법의 차이를 한 줄로 정리하면 이래요. **MC**는 "정규분포라 치고 무수히 굴려 본다", **VC**는 "정규분포라 치고 공식으로 바로 계산한다", **HS**는 "가정 없이 실제 과거 데이터에서 그냥 꺼내 쓴다"입니다. 앞 둘은 **모수적(정규 가정)**, 마지막은 **비모수적**이에요.

세 방법 모두 앞에 `np.sqrt(self.time*self.freq)`가 붙어 있죠. 이건 **시간지평 스케일링(time-horizon scaling)**, 이른바 **$\sqrt{t}$ 법칙**입니다.

$$
\text{VaR}(t\text{일}) = \sqrt{t}\;\cdot\;\text{VaR}(1\text{일})
$$

즉 하루짜리 VaR을 $t$일짜리로 늘릴 때 $t$배가 아니라 $\sqrt{t}$배를 곱해요. 수익률이 서로 독립이면 분산은 시간에 비례하고, 표준편차(변동성)는 그 제곱근에 비례하기 때문입니다.

> 📝 **예시** — 하루 VaR이 $-2\%$인데 $10$일 지평으로 늘리고 싶다면, $10$배($-20\%$)가 아니라 $\sqrt{10} \approx 3.16$배를 곱해 약 $-6.3\%$가 됩니다. 손실이 시간에 정비례해 쌓이지는 않는다는 뜻이에요.

### 롤링 윈도우 + 연속 학습(셀 14)

마지막으로 이 모든 조각을 엮어 **하루씩 앞으로 굴리며** DeepVaR와 전통 기법을 비교하는 백테스트 루프예요.

```python
for i in range(10):
    estimator = DeepARModel(prediction_length=1, context_length=15, epochs=5).fit(returns.iloc[:-10])
    predictions = list(estimator.predict(DeepARModel().list_dataset(returns.iloc[:-10+i], train=False), 1000))
    deepvar95.append(var_p(predictions, returns.iloc[-260+i:-10+i], w, 0, alpha=95))
    var99 = VaRCalculation(time=1, alpha=99).predict(returns.iloc[-260+i:-10+i], w)
```

$10$일에 걸쳐 하루씩 반복하는 루프입니다. 반복 변수 `i`가 커질수록 관찰 창이 하루씩 미끄러져요.

- `DeepARModel(prediction_length=1, ...).fit(returns.iloc[:-10])` — 매 반복마다 **모델을 새로 학습**합니다. `prediction_length=1`이니 이번엔 **딱 하루 앞**만 예측하도록 맞췄어요. 이렇게 매번 최신 데이터로 다시 배우는 방식을 **연속 학습(continuous learning)**이라 부릅니다.
- `estimator.predict(... returns.iloc[:-10+i] ..., 1000)` — 학습된 모델로 $1000$개 몬테카를로 샘플을 뽑아 예측 분포를 만듭니다. `iloc[:-10+i]`라 `i`가 커질수록 끝점이 하루씩 뒤로 밀려요.
- `var_p(predictions, returns.iloc[-260+i:-10+i], w, 0, alpha=95)` — 앞서 본 `var_p`로 $95\%$ DeepVaR을 계산해 리스트에 쌓습니다. 여기서 상관행렬 재료로 **`iloc[-260+i:-10+i]`**, 즉 최근 약 $250$일 창을 잘라 씁니다. 창이 `i`와 함께 이동하니, 이 부분만큼은 **롤링 상관**의 정신에 훨씬 가까워요.
- `VaRCalculation(time=1, alpha=99).predict(...)` — 같은 데이터 창으로 전통 기법(VC·HS)의 $99\%$ VaR도 계산합니다. 이렇게 **DeepVaR vs 전통 기법**을 같은 날짜에서 나란히 비교하는 거예요.

> 🎯 **핵심** — 이 루프의 두 얼굴을 기억하세요. 하나는 **연속 학습**(매 스텝 재학습으로 최신 시장에 적응), 다른 하나는 **롤링 윈도우**(상관 계산 창 `iloc[-260+i:-10+i]`을 하루씩 이동)예요. 다만 앞서 본 `var_p` 내부의 `returns.corr()`는 여전히 넘겨받은 창 **전체**의 단일 상관을 쓴다는 점에서, 완전한 $125$일 롤링과는 미묘하게 어긋난다는 걸 함께 알아 두면 좋습니다.

---

## 코드 완전 해설 ③ — Multi-Transformer 노트북(3_MT_GARCH.ipynb)

이제 이 프로젝트에서 **가장 흥미로운 코드**로 넘어갑니다. 앞 레슨에서 개념으로만 배운 **멀티-트랜스포머(Multi-Transformer)**, 그 핵심인 **AMH(Attention averaged Multi-Head, 논문 식 20-21)**가 실제 파이썬으로 어떻게 구현되는지 두 눈으로 확인할 거예요. 결론부터 말하면, "여러 어텐션을 평균 낸다"는 그 웅장한 아이디어가 **`for` 반복문 몇 줄**로 깔끔하게 표현됩니다. 그 대비가 이 노트북의 백미예요.

### MultiHeadSelfAttention 클래스 — 표준 어텐션 한 단위

먼저 멀티-트랜스포머의 **부품**인, 표준 트랜스포머의 어텐션 한 단위부터 봅니다.

```python
class MultiHeadSelfAttention(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads=8):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        if embed_dim % num_heads != 0:
            raise ValueError("임베딩 차원은 헤드 수로 나누어떨어져야 함")
        self.projection_dim = embed_dim // num_heads    # 헤드당 차원
        self.query_dense = tf.keras.layers.Dense(embed_dim)  # Q 사영
        self.key_dense   = tf.keras.layers.Dense(embed_dim)  # K 사영
        self.value_dense = tf.keras.layers.Dense(embed_dim)  # V 사영
        self.combine_heads = tf.keras.layers.Dense(embed_dim)# W^O
```

생성자에서 부품들을 준비합니다.

- `embed_dim` — 각 시점을 표현하는 **임베딩 차원(embedding dimension)**, 즉 벡터 하나의 길이예요.
- `num_heads=8` — **헤드(head)** 개수, 즉 어텐션을 몇 갈래로 병렬 계산할지.
- `if embed_dim % num_heads != 0:` — 임베딩 차원이 헤드 수로 **나누어떨어지지 않으면** 에러를 냅니다. 차원을 헤드별로 정확히 쪼개야 하니까요.
- `self.projection_dim = embed_dim // num_heads` — 그래서 **헤드 하나가 맡는 차원**은 전체를 헤드 수로 나눈 값입니다.
- `query_dense`, `key_dense`, `value_dense` — 입력을 각각 **쿼리 $Q$·키 $K$·값 $V$**로 사영하는 **밀집층(dense layer)** 세 개예요.
- `combine_heads` — 여러 헤드 결과를 합친 뒤 한 번 더 사영하는 층으로, 수식의 출력 가중치 $W^O$에 해당합니다.

```python
    def attention(self, query, key, value):
        score = tf.matmul(query, key, transpose_b=True)     # QKᵀ (내적=유사도)
        dim_key = tf.cast(tf.shape(key)[-1], tf.float32)    # d_k
        scaled_score = score / tf.math.sqrt(dim_key)        # ÷√d_k
        weights = tf.nn.softmax(scaled_score, axis=-1)      # softmax→확률
        output = tf.matmul(weights, value)                  # 확률·V
        return output, weights
```

이 메서드가 바로 앞 레슨에서 배운 **스케일드 닷-프로덕트 어텐션** 공식의 코드 구현이에요.

$$
\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{Q K^{\top}}{\sqrt{d_k}}\right) V
$$

각 줄을 수식과 하나하나 맞춰 볼게요.

- `score = tf.matmul(query, key, transpose_b=True)` → $Q K^{\top}$. 쿼리와 키의 **내적(dot product)**으로 둘이 얼마나 닮았는지, 즉 **유사도**를 잽니다.
- `dim_key = ... tf.shape(key)[-1]` → $d_k$. 키의 마지막 차원, 곧 **키 벡터의 차원**입니다.
- `scaled_score = score / tf.math.sqrt(dim_key)` → $\div \sqrt{d_k}$. 차원이 크면 내적이 과도하게 커져 softmax가 한쪽으로 쏠리므로, 제곱근으로 나눠 **스케일링**합니다.
- `weights = tf.nn.softmax(scaled_score, axis=-1)` → $\text{softmax}(\cdot)$. 점수들을 합이 $1$인 **확률**로 바꿉니다.
- `output = tf.matmul(weights, value)` → $(\cdot) V$. 그 확률로 값 $V$의 **가중합**을 냅니다. 이게 어텐션의 최종 출력이에요.

> 💡 **쉬운 설명** — 어렵게 느껴지면 딱 네 동작으로 외우세요. **닮은 정도 재기($QK^{\top}$) → 크기 다듬기($\div\sqrt{d_k}$) → 확률로 바꾸기(softmax) → 그 확률로 값 섞기($\times V$).** 이 네 줄이 트랜스포머의 심장입니다.

```python
    def separate_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.projection_dim))
        return tf.transpose(x, perm=[0, 2, 1, 3])
```

`separate_heads`는 하나로 뭉친 큰 벡터를 **헤드 수만큼 쪼개는** 도우미예요. `tf.reshape`로 전체 차원을 `(batch, seq, num_heads, projection_dim)`으로 나눈 뒤, `tf.transpose`로 축 순서를 바꿔 `(batch, num_heads, seq, projection_dim)`으로 만듭니다. 이렇게 헤드 축을 앞으로 빼 두면, 각 헤드가 **서로 독립적으로 병렬 어텐션**을 계산할 수 있어요.

```python
    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        query = self.query_dense(inputs); key = self.key_dense(inputs); value = self.value_dense(inputs)
        query = self.separate_heads(query, batch_size)
        key   = self.separate_heads(key, batch_size)
        value = self.separate_heads(value, batch_size)
        attention, weights = self.attention(query, key, value)
        attention = tf.transpose(attention, perm=[0, 2, 1, 3])
        concat_attention = tf.reshape(attention, (batch_size, -1, self.embed_dim))
        output = self.combine_heads(concat_attention)
        return output
```

`call`은 이 부품들을 순서대로 실행하는 **전체 흐름**이에요.

1. `batch_size = tf.shape(inputs)[0]` — 입력의 첫 차원에서 **배치 크기**를 읽습니다.
2. `query/key/value = ...dense(inputs)` — 입력을 $Q$·$K$·$V$ 세 갈래로 **선형변환**합니다.
3. `separate_heads(...)` 세 번 — 각각을 헤드별로 **분리**합니다.
4. `self.attention(query, key, value)` — 헤드별로 앞의 어텐션 공식을 계산합니다.
5. `tf.transpose(...)` 후 `tf.reshape(...)` — 헤드별 결과를 다시 원래 축 순서로 돌리고 **이어붙여(concatenate)** 하나의 큰 벡터로 합칩니다.
6. `output = self.combine_heads(concat_attention)` — 마지막으로 $W^O$($=$`combine_heads`)로 한 번 더 사영해 마무리합니다.

> 🎯 **핵심** — 여기까지가 **표준 트랜스포머의 어텐션 한 개**, 즉 논문의 **멀티-헤드(식 18-19)**입니다. $Q$·$K$·$V$ 선형변환 → 헤드 분리 → 헤드별 어텐션 → 다시 합쳐 $W^O$ 사영. 다음에 볼 `TransformerBlock`이 바로 이 부품을 **여러 개** 거느리면서 멀티-트랜스포머가 탄생해요.

### TransformerBlock 클래스 — ★멀티-트랜스포머 핵심(AMH/배깅)★

드디어 이 노트북의 하이라이트입니다. 방금 만든 어텐션 부품을 **여러 개 묶어 평균 내는** 곳이에요.

```python
class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super().__init__()
        self.Bagging = 5
        nb_dict = {}
        for i in range(self.Bagging):
            nb_dict["att" + str(i)] = MultiHeadSelfAttention(embed_dim, num_heads)
        self.nb_dict = nb_dict
        self.ffn = keras.Sequential([
            layers.Dense(ff_dim, activation="relu"),
            layers.Dense(embed_dim),
        ])
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)
```

생성자에서 가장 중요한 줄은 단연 `self.Bagging = 5`예요.

- `self.Bagging = 5` — 어텐션 유닛을 **몇 개** 둘지 정합니다. 여기선 $5$개예요.
- `for i in range(self.Bagging): nb_dict["att"+str(i)] = MultiHeadSelfAttention(...)` — 그래서 앞에서 만든 `MultiHeadSelfAttention`을 **`att0`부터 `att4`까지 다섯 개** 만들어 딕셔너리에 담습니다. 이게 논문이 말한 "여러 어텐션의 평균"의 재료예요.
- `self.ffn` — 어텐션 뒤에 붙는 **완전연결 순방향 신경망(FFN, feed-forward network)**입니다. `Dense(ff_dim, relu)`로 늘렸다가 `Dense(embed_dim)`으로 다시 줄여요.
- `layernorm1`, `layernorm2` — 두 개의 **층 정규화(layer normalization)**. 잔차 연결 뒤 값을 안정적으로 다듬습니다.
- `dropout1`, `dropout2` — 두 개의 **드롭아웃**. 과적합을 막는 동시에, 곧 볼 배깅에서 결정적 역할을 해요.

> 📝 **예시** — 표준 트랜스포머라면 이 자리에 어텐션이 딱 **$1$개**만 있어요(`4_T_GARCH` 노트북이 그렇습니다). 그런데 이 멀티-트랜스포머는 `att0`~`att4`, 무려 **$5$개**를 준비합니다. 이 개수 하나의 차이가 T와 MT를 가르는 전부예요.

```python
    def call(self, inputs, training):
        attn_output = 0
        for i in range(self.Bagging):
            aux = self.dropout1(inputs, training=training)
            attn_output = attn_output + self.nb_dict["att" + str(i)](aux) / self.Bagging
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)
```

이 `call` 안의 **`for` 반복문**이 바로 논문 **식 20-21**, 즉 **AMH**의 정체입니다. 수식으로 쓰면 이래요.

$$
\text{AMH}(X) = \frac{1}{n}\sum_{i=1}^{n} \text{MultiHead}_i(X), \qquad n = 5
$$

기호를 풀면, $X$는 입력, $\text{MultiHead}_i(X)$는 $i$번째 어텐션 유닛(`att`$i$)의 출력, $n=5$는 어텐션 개수(`self.Bagging`), $\frac{1}{n}\sum$은 그 $5$개를 **평균**낸다는 뜻이에요. 반복문을 한 줄씩 대응시켜 볼게요.

- `attn_output = 0` — 누적할 그릇을 $0$으로 시작합니다($\sum$의 초깃값).
- `aux = self.dropout1(inputs, training=training)` — **매 반복마다 입력에 드롭아웃을 새로 겁니다.** 그래서 `att0`이 보는 입력과 `att1`이 보는 입력이 **서로 조금씩 달라져요**. 이게 곧 **배깅(bagging, bootstrap aggregating)** 효과입니다 — 서로 다른(교란된) 데이터로 여러 학습기를 만드는 것.
- `attn_output = attn_output + self.nb_dict["att"+str(i)](aux) / self.Bagging` — $i$번째 어텐션의 출력을 **미리 $5$로 나눠서** 누적합니다. $5$개를 다 더하면 결국 $\frac{1}{5}\sum$, 즉 **평균**이 되죠.

이렇게 얻은 평균 어텐션 뒤로는 **표준 트랜스포머와 똑같은 뒷단**이 이어집니다.

- `out1 = self.layernorm1(inputs + attn_output)` — 입력을 그대로 더하는 **잔차 연결(residual connection)** 뒤 층 정규화. 수식으로 $\text{out}_1 = \text{LayerNorm}(X + \text{AMH}(X))$.
- `ffn_output = self.ffn(out1)` 후 `return self.layernorm2(out1 + ffn_output)` — FFN을 통과시키고 다시 잔차 연결 + 정규화. 수식으로 $\text{output} = \text{LayerNorm}(\text{out}_1 + \text{FFN}(\text{out}_1))$.

전체 블록을 두 줄의 수식으로 정리하면 이렇습니다.

$$
\text{out}_1 = \text{LayerNorm}\big(X + \text{AMH}(X)\big)
$$

$$
\text{output} = \text{LayerNorm}\big(\text{out}_1 + \text{FFN}(\text{out}_1)\big)
$$

> 🎯 **핵심** — 표준 트랜스포머는 이 자리에 **단일 어텐션** 하나를 놓고, 멀티-트랜스포머는 **$5$개를 평균 낸 AMH**를 놓습니다. **딱 이 한 가지**가 T와 MT를 가르는 전부예요. 나머지 잔차 연결·FFN·정규화는 완전히 똑같습니다.

> 💡 **쉬운 설명** — 그럼 왜 굳이 평균을 낼까요? 트랜스포머 하나는 무작위 초기화와 드롭아웃 탓에 **예측의 분산(variance)이 큽니다.** 들쭉날쭉하다는 뜻이죠. 그런데 서로 조금씩 다른 여러 개를 **평균**내면, 튀는 값들이 서로 상쇄되어 **분산이 줄고 과적합에도 강해집니다.** 이게 **앙상블 학습(ensemble learning)**의 원리이고, 논문이 위기 국면에서 MT가 더 안정적인 이유로 꼽는 대목이에요.

### GARCH 보조 함수 요약

이 노트북에는 신경망 말고도, 비교 대상인 전통 **GARCH** 계열과 데이터 준비를 위한 도우미 함수들이 함께 들어 있어요. 코드가 길어 핵심만 정리합니다.

- **`ReturnCalculation`** — 가격을 **로그수익률(log return)**로 바꿉니다. 수식으로 $r_t = \ln\!\left(\dfrac{P_t}{P_{t-1}}\right)$이에요. 여기서 $P_t$는 $t$시점 가격, $\ln$은 자연로그입니다. 로그수익률은 시간에 걸쳐 더하기 쉬워 금융에서 즐겨 씁니다.
- **`SDCalculation` / `TrueSDCalculation`** — 신경망이 맞혀야 할 **정답 라벨**인 **실현 변동성(realized volatility)**을 계산합니다. 일정 구간 수익률의 표준편차로 "그 기간 실제 변동성"을 만들어 학습 타깃으로 삼아요.
- **`DatabaseGeneration`** — **슬라이딩 윈도우(sliding window)**로 데이터를 훑으며, 과거 며칠(입력)과 그 다음 변동성(타깃) 쌍을 잔뜩 만들고 **학습/검증/테스트로 분할**합니다.
- **6개 GARCH 적합 함수** — `GARCH_Model_Student`, `GJR_GARCH_Model_Student`, `TARCH_Model_Student`, `EGARCH_Model_Student`, `AVGARCH_Model_Student`, `FIGARCH_Model_Student`. 모두 오차가 **스튜던트 $t$ 분포**를 따른다고 가정하는 전통 변동성 모델들로, 신경망의 성능을 견줄 **벤치마크**예요.
- **`Transformer_Model`** — 앞서 만든 `TransformerBlock`을 케라스 모델로 **조립**합니다. 흐름은 **입력 → 위치 인코딩(또는 LSTM) → TransformerBlock → 변동성 출력**이에요.

> ⚠️ **주의** — 여섯 GARCH 함수 이름 끝에 붙은 `_Student`를 놓치지 마세요. 모두 정규분포가 아니라 **스튜던트 $t$ 분포**를 가정한다는 표시예요. $t$ 분포는 꼬리가 두꺼워, 금융 수익률에 흔한 **극단적 급등락**을 정규분포보다 잘 담아냅니다.

### 6개 노트북 ↔ 논문 모델 매핑

이 프로젝트는 총 여섯 개의 노트북으로 서로 다른 모델을 비교해요. 헷갈리기 쉬우니 표로 한눈에 정리합니다.

| 노트북 | 모델 | 특징 |
| --- | --- | --- |
| `1_ANN_ARCH` | **ANN** | 기본 신경망 벤치마크 |
| `2_LSTM_GARCH` | **LSTM** | 순환신경망 벤치마크 |
| `4_T_GARCH` | **T**(단일 트랜스포머) | $\text{Bagging}=1$ (평균 없음) |
| `3_MT_GARCH` | **MT**(멀티-트랜스포머) | $\text{Bagging}=5$ (AMH 평균) |
| `6_TL_GARCH` | **TL** | T $+$ LSTM 층 |
| `5_MTL_GARCH` | **MTL** | MT $+$ LSTM 층 |

이 여섯 모델은 **두 개의 축**으로 이해하면 깔끔해요.

- **축 ①: T vs MT** — 어텐션이 **$1$개**냐(`Bagging=1`), **$5$개 평균**이냐(`Bagging=5`, AMH). 방금 본 그 `self.Bagging` 숫자 하나의 차이예요.
- **축 ②: $+$ L(LSTM) 유무** — 뒤에 **LSTM 층**을 붙이느냐. LSTM을 붙이면 **위치 인코딩(positional encoding)이 필요 없어집니다.** 순서 정보를 LSTM이 알아서 담당하기 때문이에요.

> 🎯 **핵심** — 논문의 결론은 이렇습니다. **MT·MTL이 T·TL과 전통 GARCH를 대부분의 오차지표에서 능가**했고, 특히 **위기(고변동) 국면에서 더 안정적**이었어요. 그 원동력이 바로 우리가 코드로 확인한 **`Bagging=5` 평균, 곧 AMH**입니다. "여러 번 시도해 평균 낸다"는 한 가지 철학이 헤드 단위(멀티-헤드)에서 모델 단위(멀티-트랜스포머)까지 그대로 반복된 셈이죠.
