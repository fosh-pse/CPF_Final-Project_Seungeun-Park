# MScFE 660 위험관리 (Risk Management)
# 그룹 워크 프로젝트 #1 — 완역본

> 원문: *MScFE 660 Risk Management — Group Work Project #1* (WorldQuant University, © 2022)
> 본 문서는 과제 안내문 원문(영문 5쪽)을 한국어로 완역한 것입니다. 원문의 구조·항목·표현을 빠짐없이 옮겼습니다.

---

## 채점 루브릭
> 채점 루브릭은 [여기](원문 링크)에서 확인하세요.

---

## 시나리오 (Scenario)

위험관리(Risk Management)의 3개 그룹 워크 프로젝트(GWP)에는 두 가지 목표가 있습니다.

1. **베이지안 네트워크(Bayesian nets) 예제를 직접 풀어보는 것**
2. **여러분의 캡스톤 프로젝트(capstone project)를 준비시키는 것**

실제로 위험관리는 캡스톤에 들어가기 전 여러분이 수강하는 마지막 과목입니다. 미니 캡스톤 프로젝트를 단계별로 안내하면서 여러분을 준비시키겠습니다. 캡스톤 프로젝트에는 두 가지 큰 역량이 필요합니다.

1. **주제의 선정과 식별(selection and identification of a topic)**
2. **연구 역량을 작고 현실적인 과업으로 조직화하는 것(organizing your research skills into small, realistic tasks)**

3개의 GWP에서, **1단계(주제 선정)는 우리가 이미 대신 해 두었습니다.** 우리는 베이지안 네트워크와 위험관리라는 주제를 결합한 캡스톤 예제를 선정해 두었습니다. 이번 과제에서 여러분은 두 번째 역량, 즉 **미니 캡스톤 프로젝트를 단계별로 완성하는 방법**에 집중하게 됩니다.

- **GWP1을 마칠 때:** 우리가 풀려는 **구체적인 문제**와 그것을 풀기 위해 **필요한 데이터**를 철저히 이해하게 됩니다. 완성된 GWP1은 (1) 문제를 다루는 **서론(introduction) 섹션**과 (2) 사용한 **데이터에 대한 완전한 기술(description)**의 모습을 갖춥니다.
- **GWP2를 마칠 때:** 베이지안 네트워크 방법론과 모델이 단계별로 개발되는 과정에 깊이 몰입하게 됩니다. 완성된 GWP2는 **문헌 검토(literature review)**와 미니 캡스톤의 **방법론(methodology) 섹션**의 모습을 갖춥니다.
- **GWP3을 마칠 때:** **결과 및 해석(results and interpretation) 섹션**을 추가하게 됩니다. 결과 부분은 방법론을 데이터에 적용해 얻은 **발견(findings)**을 강조하고, 목표가 달성되었는지를 평가합니다. 해석 부분은 결과와 방법론을 분석·평가합니다.

세 프로젝트를 모두 완성하면, 합쳐진 세 개의 프로젝트가 하나의 **미니 캡스톤(mini-capstone)**이 됩니다.

---

## GWP 전체 개요 (Overview of GWPs)

- **GWP 1: 문제 정식화(Problem Formulation). 데이터 수집(Data Collection).**
- **GWP 2: 방법론 기술(Methodology Description). 모델 개발(Model Development).**
- **GWP 3: 결과 해석(Interpretation of Results). 모델 개선(Improving the Model).**

---

## 과업 (Tasks)

### Step 1
**개별 작업.** 각 학생은 다음 논문을 읽습니다.

> Alvi, Danish A. *Application of Probabilistic Graphical Models in Forecasting Crude Oil Price.* 2018. University College London, Dissertation.
> https://arxiv.org/abs/1804.10869
> (제목 번역: 「원유 가격 예측에서의 확률 그래프 모델 응용」)

### Step 2
**그룹 작업.** 팀이 논의하여 **2~3쪽 분량의 문제 정식화(problem formulation)**를 작성합니다.

a. 이 학생의 학위논문이 풀려는 **문제는 무엇인가?**
b. **베이지안 네트워크가 이 문제를 푸는 데 왜 적합한가?**
c. 이 문제에 이 방법론을 사용했을 때의 **장점은 무엇인가?**

### Step 3
전 세계 유가에 영향을 주는 **거시경제(macroeconomic)·미시경제(microeconomic)·지정학(geopolitical)** 변수를 **식별·임포트·구조화·그래프화**합니다. 각자 한 그룹의 데이터를 배정받습니다.

**데이터 입수처(websites):**
- https://finance.yahoo.com/
- https://fred.stlouisfed.org/
- https://www.sedar.com/search/search_form_pc_en.htm
- https://www.eia.gov/opendata/
- https://data.world/datasets/oil

- **거시경제 데이터:** 국가·정부에 관한 데이터에 집중
- **미시경제 데이터:** 기업·개인의 의사결정에 관한 데이터
- **금융 데이터:** 증권·시장에 관한 데이터

각자 자기 데이터를 식별·임포트·구조화·그래프화할 책임이 있습니다. 역할은 다음과 같습니다.

a. **학생 A** — 거시경제 / 지정학 전문가(macroeconomic / geopolitical specialist)
b. **학생 B** — 미시경제 전문가(microeconomic specialist)
c. **학생 C** — 금융 전문가(financial specialist)

> **참고:** 2인 그룹은 2개 선택지 중 하나를 제외해야 합니다 (예: 미시경제 데이터 생략).

### Step 4
**그룹 작업.** 팀은 사용한 데이터의 **사전(dictionary)**과, **데이터·빈도(frequency)·출처(source)·시작일(start date)·종료일(end date)** 및 기타 관련 필드를 보여주는 **표(table)**를 작성합니다.

### Step 5
**데이터 정제(Clean the data).** 각자 논문의 **4.1절**을 꼼꼼히 읽습니다. 각자 정제의 한 부분을 담당합니다. 학생 A는 자신의 담당 작업을 **모든 데이터**(거시·미시·지정학)에 대해 수행합니다. 마찬가지로 학생 B도 자기 담당 작업을 **모든 데이터**에, 학생 C도 자기 담당 작업을 **모든 데이터**에 수행합니다. 이렇게 하면 각자 수집된 전체 데이터를 볼 기회를 갖게 됩니다.

a. **학생 A** — **"극단적 이상치(extreme outlier)"** 정제 담당. 나머지 데이터에 비해 극단적인 값을 식별합니다.
b. **학생 B** — **"불량 데이터(bad data)"** 정제 담당. 틀렸거나, 의심스럽거나, 중복된 값을 식별합니다.
c. **학생 C** — **"결측값(missing values)"** 정제 담당. 결측값이 있을 때 대치(imputation)·보간(interpolation)·시뮬레이션(simulation) 또는 기타 방법을 사용해 합리적인 값으로 대체합니다.

### Step 6
**그룹 작업.** 모든 정제 방법을 결합하여 데이터의 **"멸균(sterilized)" 버전**을 만듭니다. 어떤 데이터 포인트/사건/기간이 모델 데이터에서 **제거되었는지 그 이유**를 그룹이 협업하여 작성합니다.

### Step 7
데이터에 대해 **탐색적 데이터 분석(EDA)**을 수행합니다. 각자 **모든 데이터셋**을 다룹니다.

a. **학생 A** — **"분포(distributional)" 플롯** 작성
b. **학생 B** — **"시계열(time series)" 플롯** 작성
c. **학생 C** — **"다변량(multivariate)" 플롯**(분포형 및/또는 시계열형) 작성

### Step 8
**그룹 작업.** 정제된 데이터와 플롯에 대해 다음 질문에 답합니다.

a. 유가는 다른 자산 가격과 무엇이 다르게 보이는가? (예: 급등(spikes), 군집 변동성(clustered volatility), 계절성(seasonality) 등)
b. 유가 수익률(oil returns)은 어떤 종류의 분포를 갖는가?
c. 유가 수익률은 어떤 종류의 자기상관(autocorrelation)을 갖는가?
d. 유가에 대해 말할 수 있는 다른 정형화된 사실(stylized facts)은 무엇인가?

### Step 9
**모델(Model).** 각 학생은 자신의 주제에 대해 **2쪽 분량의 요약**을 작성합니다. 모든 세부사항이 아니라 **큰 그림(big picture)의 개관**을 반드시 제공하세요. **2쪽 제한은 엄격히 적용됩니다.**

a. **학생 A** — **확률 그래프 모델(probabilistic graphic models)**을 정의합니다. **신념망(belief networks)**과 **마르코프 네트워크(Markov networks)**를 반드시 구별하세요.
b. **학생 B** — **모수 학습(parameter learning)**을 설명하고 **구조 학습(structure learning)**과 구별합니다.
c. **학생 C** — **마르코프 연쇄(Markov chains)**와 **마르코프 블랭킷(Markov blankets)**을 기술합니다.

### Step 10
**그룹 작업.** 팀은 **알고리즘 1: 추론된 인과성(Algorithm 1: Inferred Causality)**의 자체 의사코드(pseudocode) 버전을 작성합니다. 이 단계에서는 Step 9의 조각들을 결합해야 합니다 (예: 추론된 인과성 알고리즘을 설명하려면 마르코프 블랭킷이 필요함).

---

## 제출 요건 및 형식 (Submission Requirements and Format)

한 명의 팀원이 그룹 전체를 대표하여 다음을 제출합니다.

1. **코드를 제외한** 모든 서술을 담은 **PDF 문서 1개** *
   - 이것은 미니 캡스톤의 **서론(introduction)**의 일부가 됩니다. 단순 질의응답(Q&A) 형식이 아니라 **응집력 있고 잘 조직된 글**이 되도록 하세요.
   - 제공된 **리포트 템플릿(Report Template)**을 사용하고 첫 페이지의 필수 정보를 채우세요.

2. 다음을 포함한 **압축 폴더(zipped folder):**
   - 코드, 그 출력, 각 질문에 대한 답을 포함한 **실행 가능한 주피터 노트북(Jupyter notebook)** **
   - 위 주피터 노트북의 **복제본(PDF 또는 HTML 형식)**. 코드의 출력을 포함하려면 PDF로 내려받기 **전에** 반드시 코드를 실행해야 합니다.

> \* 협업에는 Google Docs를 사용하세요. Course Overview에 제공된 Report Template을 업로드하는 것으로 시작합니다. 리포트가 완성되면 File → Download → PDF Document (.pdf)를 클릭하여 제출본을 받으세요.
>
> \** 실행 가능한 파이썬 프로그램을 협업으로 완성할 때는 Google Colab 또는 GitHub를 사용하세요.
>
> **주의(NOTE):** PDF는 다른 파일들을 담은 압축 폴더와 **별도로** 업로드해야 합니다. 이렇게 해야 Turnitin이 유사도 보고서(similarity report)를 생성할 수 있습니다.

---

## 채점 루브릭 (Rubric)

지도교수는 다음 루브릭으로 GWP1 그룹 제출물을 평가합니다.

| 평가 영역 | 배점 |
|---|---|
| **정량 분석 (개방형 질문, Quantitative Analysis — Open-Ended Questions)** | **40점** |
| **기술적/비기술적 보고서 (Technical and Non-Technical Reports)** | **30점** |
| **작성 및 형식 (Writing and Formatting)** | **20점** |

### ① 정량 분석 (40점)
그룹이 결과·공식, 그리고 이론 지식을 실제 금융 시나리오에 적용할 수 있어야 합니다. 구체적으로:
- 주장을 뒷받침하는 데 **필요한 모든 정보**를 제공
- **그룹 토론과 연구를 반영한 주장**을 제시
- 입장을 뒷받침하고 최신 정보를 제공하기 위해 **권위 있는 참고문헌(authoritative references)** 사용
- 더 통찰력 있는 금융 의사결정을 위한 **실용적 시사점(practical takeaways)**으로 결론

### ② 기술적/비기술적 보고서 (30점)
**기술 보고서(Technical reports)는 3개 부분으로 구성됩니다.**
1. 각 질문에 대한 코드 (질문 번호를 명시적으로 기재할 것)
2. 그 코드에 대응하는 출력
3. 그 결과로부터 합리적으로 도출되는 해석 및/또는 권장 조치

> **주의:** 기술 보고서는 모델의 기술적 세부사항(이름, 추정 방법, 모수 값 등)을 포함하고, 수행한 작업에 대한 일반론은 제외합니다. 사용한 **파이썬 코드의 이름은 포함하지 말아야 합니다.**

**비기술 보고서(Non-technical reports)는 3개 부분으로 구성됩니다.**
1. 결과에 대한 명확한 설명
2. 그로부터 따라오는 권장 조치
3. 각 포트폴리오에 영향을 미치는 요인의 식별

> **주의:** 모델 이름·알고리즘·불필요한 세부사항에 대한 모든 언급을 **피하세요.** 대신 **투자 의사결정**에 집중하세요.

### ③ 작성 및 형식 (20점)
전문적으로 보이는 제출물은 다음을 갖춰야 합니다.
- **제출 요건 및 형식** 목록의 모든 항목 포함
- 그래프에 **축(axes)·라벨(labels)·눈금(scales)** 포함
- 중대한 **문법 오류나 오타가 없을 것**
- 조직적이고 잘 구조화되어 **읽기 쉬운 문서**일 것
- **MLA 형식의 올바른 인용과 참고문헌(bibliography)** 포함

---

> MScFE | © 2022 - WorldQuant University – All rights reserved.
