## 코드 완전 해설 ① — Bayer의 backtest.py

이번 장에서는 Bayer가 공개한 `backtest.py`를 한 줄 한 줄 뜯어봅니다. 이 파일은 놀랍게도 **백테스팅(backtesting)** 전용 클래스 하나, 즉 `Backtest` 클래스만으로 이루어져 있어요. 여기서 백테스팅이란 "우리가 어제 계산해 둔 위험 예측치가 오늘 실제 손익과 견주어 봤을 때 잘 맞았는가?"를 사후에 채점하는 작업입니다.

우리가 채점하려는 대상은 **VaR(Value at Risk, 밸류앳리스크)** 예측치예요. VaR는 "앞으로 하루 동안 최악의 경우 이 정도까지는 잃을 수 있다"는 손실의 경계선(분위수)입니다. 예컨대 1% VaR라면 "100일 중 1일 꼴로만 이 선을 넘는 손실이 나야 정상"이라는 뜻이죠. `Backtest` 클래스는 이 "1일 꼴"이 실제로 지켜졌는지를 세 가지 방식으로 검증합니다. 즉 (1) 초과가 몇 번 났는지 세고, (2) 여러 종류의 **손실함수(loss function)** 로 예측의 품질에 점수를 매기고, (3) Christoffersen과 Engle-Manganelli의 통계 검정으로 "초과가 무작위로 잘 흩어져 있는가"를 판정합니다. 그럼 생성자부터 순서대로 살펴볼게요.

### 생성자와 기본 메서드

```python
class Backtest:
    def __init__(self, actual, forecast, alpha):
        self.index = actual.index      # 시계열 날짜 인덱스 보관
        self.actual = actual.values    # 실제 수익률 (numpy 배열)
        self.forecast = forecast.values# VaR 예측값 (numpy 배열)
        self.alpha = alpha             # 유의수준 (예: 0.01 = 99% VaR)
```

생성자 `__init__`는 객체를 만들 때 딱 한 번 실행되어, 앞으로 계속 쓸 재료 네 가지를 인스턴스 안에 챙겨 둡니다.

- `self.index`에는 `actual.index`, 즉 날짜(시계열 인덱스)만 따로 보관합니다. 나중에 그래프의 가로축으로 쓰기 위해서예요.
- `self.actual`에는 `actual.values`를 담습니다. `.values`를 붙이면 pandas 시리즈에서 알맹이 숫자만 뽑아 **넘파이 배열(numpy array)** 로 바꿔 줍니다. 이렇게 하면 뒤에서 벡터 연산을 빠르게 할 수 있어요. 이건 실제로 관측된 수익률입니다.
- `self.forecast`에는 마찬가지로 VaR 예측값 배열을 담습니다.
- `self.alpha`에는 **유의수준(significance level)** $\alpha$를 담습니다.

여기서 규약을 꼭 짚고 넘어가야 합니다. 논문에서 흔히 "99% 신뢰수준의 VaR"라고 부르는 것을, 이 코드는 `alpha=0.01`로 표현해요. 즉 신뢰수준 $99\%$와 유의수준 $\alpha=0.01$은 $1-\alpha$ 관계로 서로 반대편에서 같은 것을 가리킵니다. $\alpha$는 "우리가 이 정도 비율만큼은 초과가 나기를 목표로 삼은 확률"입니다.

> 💡 **쉬운 설명** — $\alpha$를 "허용된 실패율"이라고 생각하면 편해요. $\alpha=0.01$이면 "$100$번 중 $1$번은 VaR 선을 넘어도 괜찮다"는 약속입니다. 신뢰수준 $99\%$가 곧 $1-\alpha=0.99$이니, 둘은 같은 이야기의 앞뒷면일 뿐이에요.

> ⚠️ **주의** — 부호 규약을 헷갈리면 코드 전체가 뒤집힙니다. 이 파일에서 $\alpha$는 신뢰수준이 아니라 그 여집합인 목표 초과 확률이에요. 그래서 "$99\%$ VaR"를 넣을 때 `alpha`에는 $0.99$가 아니라 $0.01$을 넣어야 합니다.

```python
    def hit_series(self):
        return (self.actual < self.forecast) * 1
```

`hit_series`는 이 클래스의 심장입니다. 매일매일 "초과가 났는가, 아닌가"를 $0$과 $1$로 기록한 **히트 시리즈(hit series)** 를 만들어 돌려줘요.

`self.actual < self.forecast`는 넘파이의 벡터 비교라서, 모든 날짜에 대해 한꺼번에 참/거짓(불리언) 배열을 만듭니다. 여기서 실제 수익률이 VaR 예측보다 **아래(actual < forecast)** 라는 것은 곧 "예상했던 최악의 선보다 더 나쁜 손실이 났다", 즉 **초과(exceedance)** 가 발생했다는 뜻이에요. 마지막에 `* 1`을 곱하면 `True`는 $1$로, `False`는 $0$으로 바뀝니다. 논문의 식 (7)에 해당하는 지표변수죠. 수식으로 쓰면 다음과 같습니다.

$$
I_t = \begin{cases} 1 & \text{만약}\ \ \text{actual}_t < \text{forecast}_t \\ 0 & \text{그 외} \end{cases}
$$

여기서 $I_t$는 시점 $t$의 히트(초과 여부)이고, $\text{actual}_t$는 그날 실제 수익률, $\text{forecast}_t$는 그날의 VaR 예측치입니다. VaR는 보통 음수(손실 쪽)이기 때문에, 실제 수익률이 그보다 더 내려가면 $I_t=1$이 됩니다.

> 📝 **예시** — 어느 날 VaR 예측이 $\text{forecast}_t = -0.03$(즉 $-3\%$)이었다고 합시다. 실제 수익률이 $\text{actual}_t = -0.05$였다면 $-0.05 < -0.03$이 참이므로 $I_t = 1$, 초과입니다. 반대로 $\text{actual}_t = -0.01$이었다면 $-0.01 < -0.03$은 거짓이라 $I_t = 0$, 안전하게 넘긴 날이에요.

```python
    def number_of_hits(self):
        return self.hit_series().sum()      # 총 초과 횟수 (식 9)
    def hit_rate(self):
        return self.hit_series().mean()     # 초과 비율 = 위반율 (식 10)
    def expected_hits(self):
        return self.actual.size * self.alpha# 기대 초과 수 = N × alpha (식 8)
```

이 세 메서드는 히트 시리즈를 요약하는 기본 통계량입니다.

- `number_of_hits`는 `hit_series().sum()`, 즉 $1$의 개수를 모두 더해 **총 초과 횟수**를 셉니다. 식 (9)의 $\sum_{t=1}^{N} I_t$예요.
- `hit_rate`는 `.mean()`으로 평균을 냅니다. $0$과 $1$로만 이루어진 배열의 평균은 곧 "$1$이 차지하는 비율"과 같으므로, 이 값이 바로 실제 관측된 초과 비율, 즉 **위반율(violation rate)** 입니다. 식 (10)의 $\hat\pi = \frac{1}{N}\sum_{t=1}^{N} I_t$죠.
- `expected_hits`는 `self.actual.size * self.alpha`로, 표본 크기 $N$(전체 날짜 수)에 목표 확률 $\alpha$를 곱합니다. "이론적으로라면 초과가 몇 번쯤 나야 하는가"를 뜻하는 **기대 초과 수** $N\times\alpha$이고, 식 (8)에 해당합니다.

> 🎯 **핵심** — 백테스팅의 첫 관문은 이 두 숫자를 비교하는 것입니다. 실제 초과 수 $\sum_t I_t$가 기대치 $N\alpha$와 비슷해야 VaR 모형이 "옳은 개수"를 맞힌 것이에요. 예를 들어 $N=1000$이고 $\alpha=0.01$이면 기대 초과 수는 $N\alpha = 10$번이고, 실제 위반율 $\hat\pi$는 $0.01$ 언저리여야 합니다.

```python
    def duration_series(self):
        hit_series = self.hit_series()
        hit_series[0] = 1
        hit_series[-1] = 1
        return np.diff(np.where(hit_series == 1))[0]
```

`duration_series`는 초과와 초과 사이가 며칠씩 벌어져 있는지, 그 **간격(duration)** 을 계산합니다. 초과가 여기저기 골고루 흩어져 있으면 간격들이 들쭉날쭉 다양하겠지만, 만약 초과가 특정 시기에 우르르 몰려 있다면 간격이 짧은 값들이 뭉쳐서 나올 거예요. 그러니 이 배열은 "초과들이 서로 독립적으로 흩어져 있는지"를 눈으로 확인하는 진단 도구입니다.

한 줄씩 볼게요. 먼저 `hit_series()`로 히트 배열을 새로 받아옵니다. 그다음 `hit_series[0] = 1`과 `hit_series[-1] = 1`로 배열의 맨 앞과 맨 뒤를 강제로 $1$로 만들어요. 이건 "관측 시작·끝을 초과가 난 것처럼 취급"해서 첫 간격과 마지막 간격도 계산에 넣기 위한 처리(경계 보정)입니다. 이어서 `np.where(hit_series == 1)`은 값이 $1$인 위치(인덱스)들을 찾아 주고, `np.diff(...)`는 그 인접한 위치들의 차이를 구합니다. 즉 초과가 난 날들의 간격이죠. 마지막 `[0]`은 `np.where`가 튜플 형태로 결과를 돌려주기 때문에 그 안의 실제 배열을 꺼내는 것입니다.

> ⚠️ **주의** — `hit_series[0]`과 `hit_series[-1]`에 값을 직접 대입하는 것은 배열을 그 자리에서 바꿔 버리는 동작이에요. 다행히 `self.hit_series()`가 호출될 때마다 새 배열을 만들어 주므로 원본 데이터가 오염되지는 않습니다. 하지만 이런 식의 자리 바꾸기는 습관적으로 조심해야 합니다.

### 손실함수 4종

지금까지는 "초과가 났느냐 안 났느냐"만 셌지만, 손실함수는 한 걸음 더 나아가 "얼마나 잘, 혹은 얼마나 아슬아슬하게 맞혔는가"에 연속적인 점수를 매깁니다. 여기 나오는 네 함수는 모두 기본값 `return_mean=True`를 가지는데, 참이면 날짜별 손실을 평균 내어 하나의 대표 점수로 돌려주고, 거짓이면 날짜별 손실 배열을 통째로 돌려줍니다. 앞으로 편의상 그날의 예측 오차(실제와 예측의 차이)를 $m_t = \text{actual}_t - \text{forecast}_t$라 부를게요.

```python
    def tick_loss(self, return_mean=True):
        loss = (self.alpha - self.hit_series()) * (self.actual - self.forecast)
```

**틱 손실(tick loss)** 은 분위수 회귀에서 쓰는 그 유명한 핀볼(pinball) 손실입니다. 식 (13)에 해당하고, 수식은 다음과 같아요.

$$
L_t^{\text{tick}} = (\alpha - I_t)\,(\text{actual}_t - \text{forecast}_t)
$$

$\alpha$는 목표 확률, $I_t$는 히트, 뒤의 괄호는 예측 오차 $m_t$입니다. 이 손실이 왜 영리한지 볼게요. 초과가 안 난 날($I_t=0$)에는 계수가 $\alpha$(작은 양수)이고 보통 $m_t>0$(실제가 VaR 위)이라 작은 벌점이 붙습니다. 반대로 초과가 난 날($I_t=1$)에는 계수가 $\alpha-1$(음수)인데 그날은 $m_t<0$이라, 음수 곱하기 음수로 다시 양수가 되고 가중치는 $1-\alpha$(큰 값)로 커집니다. 즉 초과를 훨씬 무겁게 벌하죠. 이 손실의 기댓값을 최소로 만드는 예측치가 바로 참된 $\alpha$-분위수라서, VaR 예측의 정확도를 재는 데 이론적으로 딱 맞습니다.

> 💡 **쉬운 설명** — 틱 손실은 "안전할 때는 살짝, 뚫렸을 때는 크게" 벌점을 주는 저울이에요. $\alpha$가 작을수록 초과 한 번의 벌점 가중치 $1-\alpha$가 커져서, 드물어야 할 사건을 놓쳤을 때 더 크게 혼내는 구조입니다.

```python
    def smooth_loss(self, delta=25, return_mean=True):
        loss = ((self.alpha - (1 + np.exp(delta*(self.actual - self.forecast)))**-1)
                * (self.actual - self.forecast))
```

**부드러운 손실(smooth loss)** 은 방금 본 틱 손실의 매끄러운 버전입니다. 식 (12)에 해당해요. 문제는 히트 $I_t$가 $m_t$의 부호에 따라 $0$에서 $1$로 뚝 끊기는 계단함수라 미분이 안 된다는 점입니다. 그래서 이 계단을 **시그모이드(sigmoid)** 함수로 부드럽게 이어 줍니다.

$$
L_t^{\text{smooth}} = \left(\alpha - \frac{1}{1 + e^{\delta\, m_t}}\right) m_t, \qquad m_t = \text{actual}_t - \text{forecast}_t
$$

코드의 `(1 + np.exp(delta*(...)))**-1`이 바로 $\dfrac{1}{1+e^{\delta m_t}}$입니다. `**-1`은 역수(즉 $\cdots^{-1}$)를 뜻해요. 이 시그모이드는 $m_t$가 크게 음수면(강한 초과) $1$에 가까워지고, 크게 양수면 $0$에 가까워져서 히트 $I_t$의 역할을 매끄럽게 흉내 냅니다. $\delta$는 계단의 가파르기를 조절하는 손잡이로, 기본값 `delta=25`는 꽤 날카롭게 세팅한 값이에요.

> 🎯 **핵심** — $\delta$를 무한대로 키우면 시그모이드가 진짜 계단함수가 되어 부드러운 손실은 틱 손실과 완전히 같아집니다. 즉 부드러운 손실은 "미분 가능해서 최적화·비교에 편리한 틱 손실"이라고 이해하면 됩니다. $\delta=25$는 그 근사를 실용적으로 충분히 가파르게 만든 타협점이에요.

```python
    def quadratic_loss(self, return_mean=True):
        loss = (self.hit_series() * (1 + (self.actual - self.forecast)**2))
```

**이차 손실(quadratic loss)** 은 식 (11)로, 초과가 난 날에만 집중해서 그 심각도를 재는 손실입니다.

$$
L_t^{\text{quad}} = I_t\left(1 + (\text{actual}_t - \text{forecast}_t)^2\right) = I_t\left(1 + m_t^2\right)
$$

앞에 $I_t$가 곱해져 있어서 초과가 안 난 날($I_t=0$)은 손실이 $0$입니다. 초과가 난 날에는 $1$에 오차의 제곱 $m_t^2$을 더한 값이 벌점이 되죠. 제곱이라 초과 폭이 두 배가 되면 벌점은 대략 네 배로 커집니다. 그래서 이차 손실은 "얼마나 자주 뚫렸나"보다 "뚫렸을 때 얼마나 크게 뚫렸나"를 중시하는 손실이에요.

> 📝 **예시** — 같은 초과라도 오차 $m_t=-0.01$일 때는 벌점이 $1 + 0.0001 = 1.0001$이지만, $m_t=-0.10$일 때는 $1 + 0.01 = 1.01$입니다. 큰 붕괴가 났을 때 제곱항이 훨씬 크게 반응하는 게 이 손실의 성격이에요.

```python
    def firm_loss(self, c=1, return_mean=True):
        loss = (self.hit_series() * (1 + (self.actual - self.forecast)**2)
                - c*(1-self.hit_series()) * self.forecast)
```

**펌 손실(firm loss)** 은 식 (14)로, 금융회사(firm)의 현실적 고민을 담은 손실입니다. 앞부분은 방금 본 이차 손실 그대로이고, 여기에 항 하나가 더 붙어요.

$$
L_t^{\text{firm}} = I_t\left(1 + m_t^2\right) - c\,(1 - I_t)\,\text{forecast}_t
$$

새로 붙은 $-c(1-I_t)\,\text{forecast}_t$ 항을 뜯어보면, $1-I_t$ 때문에 이 항은 초과가 안 난 날($I_t=0$)에만 살아납니다. 그런데 VaR 예측치 $\text{forecast}_t$는 보통 음수이므로 $-\text{forecast}_t$는 양수가 되고, 결국 "초과가 안 난 날마다 붙는 양의 비용"이 됩니다. 이건 VaR를 너무 보수적으로(너무 크게) 잡아 두면 그만큼 자본을 묶어 둬야 하는 **기회비용(opportunity cost)** 을 뜻해요. 계수 `c`(논문에서는 $a$)는 이 자본 비용의 무게이고 기본값은 `c=1`입니다.

> 💡 **쉬운 설명** — 펌 손실은 회사 입장의 두 가지 고민을 한 저울에 올린 것입니다. 하나는 "뚫리면 위험하다"(이차 손실 부분), 다른 하나는 "너무 안전하게만 잡으면 돈이 놀아서 손해다"(자본 비용 부분). $c$를 키우면 자본을 아끼라는 압력이 세지고, 줄이면 안전을 더 중시하게 됩니다.

### lr_bt — Christoffersen 우도비 검정

이제 이 파일의 백미인 `lr_bt`입니다. Christoffersen이 제안한 **우도비 검정(likelihood ratio test)** 으로, 좋은 VaR라면 초과가 (1) 올바른 개수만큼 나야 하고 **동시에** (2) 서로 뭉치지 않고 독립적으로 흩어져 있어야 한다는 두 조건을 함께 검사합니다. 조각별로 따라가 볼게요.

```python
    def lr_bt(self):
        hits = self.hit_series()
        tr = hits[1:] - hits[:-1]      # 인접한 날의 차이로 '전이(transition)' 탐지
```

먼저 히트 배열을 받아 `hits`에 담고, `hits[1:] - hits[:-1]`로 하루씩 어긋나게 뺀 차이 배열 `tr`을 만듭니다. `hits[1:]`은 둘째 날부터 끝까지, `hits[:-1]`은 첫째 날부터 끝에서 둘째까지라서, 둘을 빼면 "오늘 상태 빼기 어제 상태"가 되는 거예요. 이게 상태 사이의 **전이(transition)** 를 잡아냅니다. `tr`이 $+1$이면 어제 $0$(비초과)에서 오늘 $1$(초과)로 바뀐 것($0\to1$), $-1$이면 $1\to0$, $0$이면 어제와 오늘 상태가 같은 것(유지)입니다.

```python
        n01, n10 = (tr == 1).sum(), (tr == -1).sum()
        n11, n00 = (hits[1:][tr == 0] == 1).sum(), (hits[1:][tr == 0] == 0).sum()
```

이제 네 종류의 전이 횟수를 셉니다. `n01`은 `tr == 1`의 개수, 즉 비초과에서 초과로 넘어간 횟수($0\to1$)입니다. `n10`은 `tr == -1`의 개수, 초과에서 비초과로 돌아온 횟수($1\to0$)예요. 나머지 둘은 상태가 유지된 날들(`tr == 0`) 중에서 나눕니다. `hits[1:][tr == 0]`는 "어제와 상태가 같은 날들의 오늘 값"인데, 그중 값이 $1$이면 `n11`(초과 다음날도 초과, $1\to1$), $0$이면 `n00`(비초과 다음날도 비초과, $0\to0$)입니다. 특히 `n11`은 초과가 연달아 붙는 "뭉침 정도"를 나타내는 핵심 숫자예요.

```python
        n0, n1 = n01 + n00, n10 + n11   # 상태 0·1에 머문 총 횟수
        n = n0 + n1
        p01, p11 = n01 / (n00 + n01), n11 / (n11 + n10)  # 조건부 전이확률
        p = n1 / n                       # 전체 초과 확률(추정)
```

여기서 카운트를 확률로 바꿉니다. `n0 = n01 + n00`은 다음날이 비초과($0$)로 끝난 날의 총수, `n1 = n10 + n11`은 다음날이 초과($1$)로 끝난 날의 총수이고, `n = n0 + n1`은 전이 관측의 총 개수입니다. 그다음 세 확률을 추정해요.

- `p01 = n01 / (n00 + n01)`은 $\pi_{01}$, 즉 "어제 비초과였는데 오늘 초과할 조건부 확률"입니다.
- `p11 = n11 / (n11 + n10)`은 $\pi_{11}$, "어제 초과였는데 오늘도 초과할 조건부 확률"입니다.
- `p = n1 / n`은 $\hat p$, 어제 상태를 따지지 않은 전체 초과율입니다.

만약 초과가 정말로 서로 **독립(independence)** 이라면, 오늘 초과할 확률이 어제 상태에 좌우되지 않아야 합니다. 즉 다음이 성립해야 해요.

$$
\pi_{01} \approx \pi_{11} \approx \hat p
$$

> 🎯 **핵심** — 독립성 검정의 아이디어가 바로 이 한 줄에 담겨 있습니다. "어제 뚫렸다는 사실"이 "오늘 뚫릴 확률"을 바꾸면 안 된다는 거예요. 만약 $\pi_{11}$이 $\hat p$보다 눈에 띄게 크다면, 초과가 한번 나면 연달아 나는 뭉침(군집)이 있다는 위험 신호입니다.

```python
        if n1 > 0:
            uc_h0 = n0*np.log(1-self.alpha) + n1*np.log(self.alpha)   # 귀무: 초과율=alpha
            uc_h1 = n0*np.log(1-p) + n1*np.log(p)                     # 대립: 초과율=관측 p
            uc = -2 * (uc_h0 - uc_h1)
```

`if n1 > 0`은 초과가 한 번이라도 있어야(로그 계산에서 문제가 안 생기게) 검정을 진행한다는 안전장치입니다. 안쪽에서 먼저 **무조건부 커버리지(unconditional coverage)**, 즉 "초과 개수가 맞느냐"를 검정합니다.

- `uc_h0`은 귀무가설(초과율이 목표값 $\alpha$라는 가정) 아래의 로그우도입니다. 비초과 $n_0$번은 확률 $1-\alpha$로, 초과 $n_1$번은 확률 $\alpha$로 일어났다고 보고 $n_0\ln(1-\alpha) + n_1\ln\alpha$을 계산해요.
- `uc_h1`은 대립가설(초과율이 실제 관측값 $\hat p$라는 가정) 아래의 로그우도로, $n_0\ln(1-\hat p) + n_1\ln \hat p$입니다.
- `uc`는 두 로그우도의 차에 $-2$를 곱한 우도비 통계량입니다.

$$
LR_{uc} = -2\left[\big(n_0\ln(1-\alpha) + n_1\ln\alpha\big) - \big(n_0\ln(1-\hat p) + n_1\ln\hat p\big)\right]
$$

이 값이 크다는 것은 관측된 초과율 $\hat p$가 목표 $\alpha$와 많이 다르다는 뜻이라, VaR가 초과를 너무 많이 또는 너무 적게 냈다는 신호입니다.

```python
            ind_h0 = (n00+n01)*np.log(1-p) + (n01+n11)*np.log(p)      # 귀무: 전이가 p로 동일
            ind_h1 = n00*np.log(1-p01) + n01*np.log(p01) + n10*np.log(1-p11)
            if p11 > 0:
                ind_h1 += n11 * np.log(p11)                          # 대립: 전이확률이 상태별로 다름
            ind = -2 * (ind_h0 - ind_h1)
```

다음은 **독립성(independence)** 검정입니다.

- `ind_h0`은 귀무가설(어제 상태와 무관하게 전이확률이 모두 같은 $\hat p$라는 가정) 아래의 로그우도로, 전체를 하나의 확률 $\hat p$로 설명한 값입니다.
- `ind_h1`은 대립가설(전이확률이 상태별로 달라 $\pi_{01} \ne \pi_{11}$이라는 가정) 아래의 로그우도입니다. $n_{00}\ln(1-\pi_{01}) + n_{01}\ln\pi_{01} + n_{10}\ln(1-\pi_{11})$을 먼저 계산하고, `if p11 > 0`일 때만 $n_{11}\ln\pi_{11}$을 더합니다. $\pi_{11}=0$이면 $\ln 0$이 $-\infty$가 되어 버리니 이를 피하는 방어 코드예요.
- `ind`는 역시 두 로그우도 차에 $-2$를 곱한 우도비입니다.

$$
LR_{ind} = -2\left(\text{ind\_h0} - \text{ind\_h1}\right)
$$

이 값이 크면 상태별 전이확률이 서로 다르다는 것, 즉 초과가 직렬로 종속되어(뭉쳐서) 나타난다는 증거입니다.

```python
            cc = uc + ind    # 조건부 커버리지 = UC + IND (논문 LR_cc = LR_uc + LR_ind)
```

**조건부 커버리지(conditional coverage)** 통계량 `cc`는 앞의 둘을 그냥 더한 것입니다. Christoffersen의 아름다운 결과가 바로 이거예요. 논문 식으로 $LR_{cc} = LR_{uc} + LR_{ind}$가 성립합니다.

> 💡 **쉬운 설명** — 좋은 VaR가 되려면 두 시험을 모두 통과해야 합니다. 하나는 "초과 개수가 맞는가"($LR_{uc}$), 다른 하나는 "초과가 뭉치지 않고 흩어졌는가"($LR_{ind}$). 조건부 커버리지 $LR_{cc}$는 이 둘을 합쳐 한 번에 채점하는 종합 시험이에요.

```python
            df = pd.concat([pd.Series([uc, ind, cc]),
                pd.Series([1 - stats.chi2.cdf(uc, 1),
                           1 - stats.chi2.cdf(ind, 1),
                           1 - stats.chi2.cdf(cc, 2)])], axis=1)
```

마지막으로 세 통계량을 **카이제곱 분포(chi-squared distribution)** 를 이용해 p값으로 바꿉니다. 우도비 통계량은 표본이 클 때 근사적으로 카이제곱 분포를 따른다는 점을 쓰는 거예요. `stats.chi2.cdf(통계량, 자유도)`는 누적분포함수 값이고, $1$에서 그걸 빼면 "통계량이 이보다 클 확률", 즉 p값이 됩니다. 자유도는 $LR_{uc}$와 $LR_{ind}$가 각각 $1$, $LR_{cc}$는 $2$예요(제약이 두 개라서). `pd.concat(..., axis=1)`은 통계량 열과 p값 열을 나란히 붙여 표로 만듭니다.

$$
\text{p-value} = 1 - F_{\chi^2_k}(\text{통계량}), \quad k \in \{1, 1, 2\}
$$

여기서 $F_{\chi^2_k}$는 자유도 $k$인 카이제곱 분포의 누적분포함수입니다.

> 🎯 **핵심** — 판정 규칙은 간단합니다. **p값이 $0.05$보다 크면 통과**예요. p값이 크다는 것은 "관측된 결과가 귀무가설(VaR가 잘 맞는다)과 모순되지 않는다"는 뜻이니까요. 반대로 p값이 $0.05$보다 작으면 귀무가설을 기각하고, VaR 모형에 문제가 있다고 봅니다.

### dq_bt — Engle-Manganelli 동적 분위수 검정

`dq_bt`는 Engle과 Manganelli의 **동적 분위수 검정(dynamic quantile test)** 입니다. 아이디어는 이래요. VaR가 완벽하다면 히트 $I_t$는 과거의 그 무엇으로도 예측되지 않는 순수한 잡음(평균 $\alpha$인 독립 사건)이어야 합니다. 그래서 "히트를 과거 정보로 설명하려는 회귀"를 돌려 보고, 설명력이 정말 $0$인지를 검정하는 거죠.

```python
    def dq_bt(self, hit_lags=4, forecast_lags=1):
        hits = self.hit_series()
        p, q, n = hit_lags, forecast_lags, hits.size
        pq = max(p, q - 1)
        y = hits[pq:] - self.alpha    # 종속변수: 히트 - alpha (평균0이 이상적)
        x = np.zeros((n - pq, 1 + p + q))
        x[:, 0] = 1                   # 상수항(절편)
        for i in range(p):            # 과거 히트들을 설명변수로
            x[:, 1 + i] = hits[pq-(i+1):-(i+1)]
        for j in range(q):            # 현재+과거 VaR 예측을 설명변수로
            if j > 0:
                x[:, 1 + p + j] = self.forecast[pq-j:-j]
            else:
                x[:, 1 + p + j] = self.forecast[pq:]
```

세팅 부분을 볼게요. `hit_lags=4`는 과거 히트를 며칠치 넣을지, `forecast_lags=1`은 VaR 예측을 몇 개 넣을지 정합니다. `pq = max(p, q-1)`은 과거 값을 참조하느라 앞쪽에서 잘라내야 하는 날짜 수예요.

종속변수 `y`는 `hits[pq:] - self.alpha`, 즉 히트에서 $\alpha$를 뺀 값입니다. 수식으로 $y_t = I_t - \alpha$인데, 히트가 평균적으로 $\alpha$만큼 나야 정상이므로 잘 맞는 VaR라면 이 $y_t$의 평균은 $0$에 가깝습니다.

설명변수 행렬 `x`는 `np.zeros`로 $(n-pq)$행, $(1+p+q)$열 크기의 빈 판을 만든 뒤 채웁니다. 첫 열 `x[:, 0] = 1`은 절편(상수항)이고, 그다음 `p`개 열에는 하루씩 밀린 과거 히트들 $I_{t-1}, \dots, I_{t-p}$을, 마지막 `q`개 열에는 현재와 과거의 VaR 예측치를 넣습니다. 결국 이런 회귀식을 세우는 거예요.

$$
y_t = \beta_0 + \sum_{i=1}^{p}\beta_i\, I_{t-i} + \sum_{j=0}^{q-1}\gamma_j\, \text{forecast}_{t-j} + \varepsilon_t
$$

$\beta_0$는 절편, $\beta_i$는 과거 히트의 계수, $\gamma_j$는 VaR 예측의 계수, $\varepsilon_t$는 오차입니다. 히트가 독립이고 편향 없이 $\alpha$를 지킨다면 이 모든 계수가 $0$이어야 합니다. 즉 과거 히트도, VaR 값 자체도 오늘의 초과를 예측하는 데 아무 쓸모가 없어야 해요.

> 📝 **예시** — 만약 과거 히트의 계수 $\beta_1$이 유의하게 양수로 나온다면, "어제 뚫렸으면 오늘도 뚫릴 가능성이 높다"는 뜻이라 초과가 뭉치고 있다는 증거입니다. 마찬가지로 $\gamma_0$가 $0$이 아니면 "VaR를 크게 잡은 날일수록 초과가 더/덜 난다"는 체계적 편향이 있는 거예요.

```python
        beta = np.dot(np.linalg.inv(np.dot(x.T, x)), np.dot(x.T, y))   # (XᵀX)⁻¹Xᵀy
        lr_dq = np.dot(beta, np.dot(np.dot(x.T, x), beta)) / (self.alpha*(1-self.alpha))
        p_dq = 1 - stats.chi2.cdf(lr_dq, 1+p+q)
```

이제 계수를 실제로 풀고 검정합니다. 첫 줄은 **선형회귀(linear regression)** 의 **정규방정식(normal equation)** 그대로예요. `np.linalg.inv(x.T @ x)`가 $(X^\top X)^{-1}$, 여기에 $X^\top y$를 곱해 계수 추정치를 얻습니다.

$$
\hat\beta = (X^\top X)^{-1} X^\top y
$$

$X$는 설명변수 행렬, $X^\top$는 그 전치(transpose), $y$는 종속변수 벡터입니다. 그다음 DQ 통계량을 계산해요.

$$
DQ = \frac{\hat\beta^\top (X^\top X)\,\hat\beta}{\alpha(1-\alpha)}
$$

분자 $\hat\beta^\top(X^\top X)\hat\beta$는 추정된 계수들이 전체적으로 $0$에서 얼마나 멀리 떨어졌는지를 재는 이차형식이고, 분모 $\alpha(1-\alpha)$는 히트가 성공확률 $\alpha$인 베르누이 사건일 때의 분산이라 통계량을 표준화해 줍니다. 계수가 모두 $0$에 가까우면 분자가 작아져 $DQ$도 작아지죠. 마지막으로 이 $DQ$를 자유도 $1+p+q$인 카이제곱 분포로 검정해 p값 `p_dq`를 얻습니다.

$$
\text{p-value} = 1 - F_{\chi^2_{1+p+q}}(DQ)
$$

> 🎯 **핵심** — 여기서도 판정은 같습니다. **p값이 $0.05$보다 크면 독립(통과)** 이에요. 계수 전부가 $0$과 다를 바 없다는 뜻이니, 과거 정보로 초과를 예측할 수 없다는 좋은 결론입니다. 자유도가 $1+p+q$인 이유는 검정하는 계수가 절편 $1$개, 히트 시차 $p$개, 예측 시차 $q$개로 모두 $1+p+q$개이기 때문이에요.

> ⚠️ **주의** — $X^\top X$가 역행렬을 갖지 못하는 **특이행렬(singular matrix)** 인 경우 `np.linalg.inv`가 실패합니다. 그래서 실제 코드는 이 계산을 `try/except`로 감싸, 역행렬을 못 구하면 통계량 대신 NaN(결측)을 돌려주도록 방어해 둡니다.

### 실행 예시

```python
if __name__ == "__main__":
    Y = pd.read_csv("input/returns.txt", ...)          # IBM 일별 로그수익률
    X = pd.read_csv("input/quantile_predicitons.txt", ...)  # 여러 모델의 1% VaR 예측
    bt = Backtest(actual=Y, forecast=X.loc[:, "eGARCH-fhs"], alpha=0.01)
    bt.plot()
```

마지막은 이 파일을 직접 실행했을 때 돌아가는 예시입니다. `if __name__ == "__main__":`는 "이 파일을 (다른 데서 불러오는 게 아니라) 직접 실행할 때만" 아래 블록을 돌리라는 파이썬의 관용구예요.

- `Y`에는 `returns.txt`에서 IBM 주식의 일별 로그수익률을 읽어 옵니다.
- `X`에는 `quantile_predicitons.txt`에서 여러 모델이 각각 내놓은 1% VaR 예측치들을 읽어 옵니다.
- `Backtest(...)`로 백테스트 객체를 만드는데, `actual`에는 실제 수익률 `Y`를, `forecast`에는 그 여러 모델 중 `eGARCH-fhs` 열 하나를 골라 넣고, `alpha=0.01`로 1% VaR임을 지정합니다.
- `bt.plot()`으로 결과를 그림으로 그립니다.

정리하면, 이 예시는 IBM 주식 수익률에 대해 **eGARCH-fhs** 모형이 내놓은 1% VaR 예측을 백테스트하는 장면입니다. 지금까지 뜯어본 히트 시리즈, 손실함수 네 종, Christoffersen 검정, Engle-Manganelli 검정이 모두 이 한 객체 안에서 함께 작동하게 되는 거예요.

> 💡 **쉬운 설명** — `eGARCH-fhs`는 변동성을 다루는 eGARCH 모형에 과거 오차 분포를 재활용하는 여과 과거 시뮬레이션(filtered historical simulation, FHS) 기법을 얹은 조합입니다. 이 예시의 핵심은 "어떤 VaR 모형이든 이 `Backtest` 클래스에 넣기만 하면 똑같은 잣대로 채점된다"는 점이에요. 모형 이름만 바꿔 끼우면 여러 모형을 공정하게 비교할 수 있습니다.
