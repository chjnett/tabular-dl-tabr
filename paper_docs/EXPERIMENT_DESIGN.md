# Experiment Design and Validation

## 1. Experimental Setup

본 실험의 목적은 GateR-M 아키텍처에 새로 반영된 **Label Retrieval 활성화(버그 수정)** 및 **특성-라벨 Concat 방식(옵션 B)**의 효과를 검증하고, 기존 SOTA 모델인 TabR과의 벤치마크 비교를 수행하는 것입니다.

### 1.1 Datasets
- **California Housing (회귀):** 주택 가격 예측 (1차 성능 복원 검증용)
- **Black Friday (회귀/분류):** 사용자 소비 패턴 예측 (데이터 확장 파이프라인 검증용)

### 1.2 Evaluation Metrics
- **Regression:** Root Mean Square Error (RMSE) - 낮을수록 우수함 (Lower is better)

---

## 2. Validation Scenario 1: Performance Restoration (California)

기존 설계 결함(Label 미사용)으로 인해 GateR-M의 성능이 Test RMSE `0.5564`에 머무는 문제가 있었습니다. 이 실험은 라벨 융합 방식(옵션 B)을 적용한 새로운 HPO가 원본 TabR의 성능을 복원하거나 추월할 수 있는지 검증합니다.

* **가설:** 이웃의 라벨 정보가 텐서 흐름에 정상적으로 결합(Concat)되면, RMSE 성능은 TabR의 베이스라인인 `0.4071` 수준 이하로 진입할 것이다.
* **실험 셋업:** 
  - Optuna 기반 30 Trials HPO 실행
  - `context_size = 96` (고정 벤치마크)
  - `d_embedding` 하한선 `96` 보장
* **측정 비교:**
  - Before (버그 상태): `0.5564`
  - Target (TabR 논문 수치): `0.4071`
  - After (옵션 B 적용): HPO 최종 결과 (기대 수치 `0.40~0.42`)

---

## 3. Validation Scenario 2: Pipeline Expansion (Black Friday)

다양한 데이터 도메인에 대한 일반화(Generalization) 성능을 확인하기 위해, 신규 데이터셋 파이프라인을 구축하고 검증합니다.

* **데이터 준비 (prepare_black_friday.py):**
  - OpenML/Kaggle 등에서 Black Friday 원본 데이터를 획득
  - 수치형 데이터와 라벨을 Numpy 배열(`X_num_train.npy` 등)로 파싱
  - Train/Val/Test 비율 (60/20/20) 분할
  - `info.json` 포맷 명세 자동 생성 (TabR 호환성 보장)
* **실험 셋업:** 
  - 생성된 데이터 폴더(`data/black-friday`)를 바라보는 `gate_rm-tuning.toml` HPO 구동.

---

## 4. Ablation Study Design (분석 실험)

GateR-M에 도입된 개별 컴포넌트의 기여도를 분리하여 정량화하기 위해 다음 두 가지 Ablation 실험을 제안합니다.

### 4.1 Fusion Strategy: Add vs. Concat (Option A vs Option B)
이웃의 특성(Feature)과 라벨(Label) 임베딩을 결합하는 방식에 따른 성능 차이를 검증합니다.
- **Option A (Addition):** $V = x_{neighbors} + y_{neighbors\_emb}$
- **Option B (Concatenation):** $V = W_{fusion} \cdot [ x_{neighbors} \parallel y_{neighbors\_emb} ]$
- **검증 지표:** 수렴(Convergence)까지 걸리는 Epoch 속도 및 최종 Test RMSE 격차.

### 4.2 Network Depth and Ensemble Breadth
- **Stacked Layer 갯수 ($N$):** $N \in \{1, 2, 3, 4\}$ 변화에 따른 성능. $N=1$일 경우 단발성 정적 검색과 유사해짐. 깊은 층위의 Retrieval이 과적합 없이 어떻게 긍정적으로 작용하는지 곡선으로 시각화.
- **Packed Ensemble 브랜치 수 ($E$):** $E \in \{1, 4, 7, 10\}$ 변화에 따른 분산(Variance) 감소 효과 측정. 파라미터 추가량 대비 성능 향상 가성비(Trade-off) 산출.

---

## 5. Visualization Plan

논문(Paper) 및 리포트에 수록할 핵심 시각화 가이드라인입니다.

1. **3D Attention Map Heatmap:** 
   - X축: 피처 차원(K), Y축: 이웃 인덱스(M)
   - 색상(Intensity): Attention Softmax Score
   - 목표: 특정 노이즈 칼럼(피처) 라인 전체가 짙은 색(낮은 가중치)으로 마스킹(Cut-off)되는 시각적 증거 제시.
2. **Learning Curve Comparison:** 
   - 에포크에 따른 Validation RMSE 하락 곡선 플롯. Option A(Add)와 Option B(Concat)의 곡선 교차(Cross) 지점 및 수렴 속도 비교.
