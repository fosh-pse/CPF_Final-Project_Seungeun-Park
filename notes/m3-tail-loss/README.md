# M3 학습노트 — 딥러닝으로 꼬리손실(Tail Loss) 추정하기

VaR(Value at Risk)를 **LSTM 기반 DeepVaR**와 **트랜스포머·멀티-트랜스포머**로 추정하고,
그 결과를 **백테스팅**으로 검증하는 과정을 개념·수식·논문·코드까지 하나하나 풀어 쓴 학습노트입니다.

- 산출물(PDF): [`../M3_TailLoss_DeepLearning_StudyNote.pdf`](../M3_TailLoss_DeepLearning_StudyNote.pdf) · A4 70쪽
- 특징: 모든 수식은 **KaTeX**로 조판(깨짐 없음), 풍부한 예시·쉬운 설명 콜아웃, 코드 한 줄 단위 해설

## 구성

| 파트 | 내용 |
|---|---|
| Lesson 1–2 | DeepVaR — RNN→GRU→LSTM, DeepAR 확률예측, VaR 백테스팅(손실함수·유효성검정) |
| Lesson 3–4 | Transformer — 어텐션, 멀티-헤드, 멀티-트랜스포머(AMH/배깅) |
| 논문 ① | Fatouros et al., *DeepVaR* (Digital Finance, 2023) 완전 해설 |
| 논문 ② | Ramos-Pérez et al., *Multi-Transformer* (Mathematics, 2021) 완전 해설 |
| 코드 3종 | `backtest.py` · DeepVaR 노트북 · Multi-Transformer 노트북 |

## 다시 빌드하기

```bash
npm install
npm run pdf     # build.mjs (Markdown+KaTeX -> final.html) + print.mjs (Chromium -> PDF)
```

요구사항:
- Node 18+
- 본문 한글 렌더링을 위한 **Noto Sans CJK KR** 시스템 폰트 (예: `apt-get install fonts-noto-cjk`)
- `print.mjs`는 Chromium 실행 파일 경로를 `PW_CHROMIUM` 환경변수 또는 Playwright 기본 위치에서 찾습니다.

### 소스 구조
- `sections/01..08.md` — 파트별 원고(Markdown + `$…$`/`$$…$$` 수식)
- `build.mjs` — 섹션 병합, KaTeX 서버사이드 렌더, 콜아웃/목차/표지 생성 → `final.html`
- `style.css` — 인쇄용 스타일(콜아웃 색상, 페이지 나눔, 폰트)
- `print.mjs` — Chromium 헤드리스로 `final.html` → PDF(A4, 페이지 번호)
