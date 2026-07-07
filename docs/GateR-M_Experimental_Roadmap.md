# 🚀 GateR-M 실험 전략 로드맵 (Experimental Roadmap)

본 문서는 오리지널 TabR 대비 **GateR-M 아키텍처의 3가지 핵심 무기(피처 노이즈 필터링, 다단 추상화, 앙상블 효율성)**를 입증하기 위한 단계별 데이터셋 검증 전략입니다. 현재 진행 중인 California Housing (CA) 실험 종료 후 순차적으로 진행됩니다.

---

## 🎯 1단계: 정면 돌파 (Fair Comparison)
오리지널 TabR의 메인 성능을 입증했던 데이터셋을 활용하여, 동일한 통제 환경에서 GateR-M의 우수성을 증명합니다.

*   **California Housing (CA)** 
    *   **특징:** 수치형 피처 위주($Num=8$), 지리적 좌표 정보 포함. 이웃(Nearest Neighbor) 검색 기법이 매우 잘 통하는 대표적 데이터셋.
    *   **목표:** 오리지널 TabR 점수(`0.400`) 돌파. (현재 진행 중 🏃‍♂️)
*   **Black Friday (BL) 또는 Adult (AD)** 
    *   **특징:** 범주형(Categorical)과 이진(Binary) 피처 혼합.
    *   **목표:** 텐서 차원을 피처 단위로 쪼개는 **Feature-Wise 어텐션**이 이질적인 피처 타입에서도 노이즈를 잘 발라내는지 검증.

---

## 🌟 2단계: 치트키 검증 (Robustness against Noise)
GateR-M의 논문 킬러 파트. 피처가 많은 환경 및 극단적인 노이즈 상황에서 원래 TabR은 무너지지만, GateR-M은 살아남음을 증명합니다.

*   **Otto Group Product (OT) 또는 Shifts Weather (WE)** 
    *   **특징:** 피처 수가 매우 많음 (OT: 93개, WE: 118개). 무의미한 노이즈 컬럼이 섞여 들어올 확률이 높음.
    *   **🔥 킬러 실험 세팅 (Dummy Features 추가):**
        *   의도적으로 무작위 가우시안 노이즈 컬럼을 10개, 20개씩 강제로 추가.
        *   **가설:** 행 단위로 유사도를 계산하는 TabR은 유사도 오염으로 성능 폭락. 반면, GateR-M은 **Softmax 컷오프**를 통해 노이즈 컬럼의 가중치를 0으로 밀어버리며 성능 방어.

---

## ⚡ 3단계: 가성비 및 효율성 증명 (Parameter-Efficient Ensemble)
트리 기반 모델(XGBoost 등)이 지배적인 데이터셋에서 **Packed Ensemble**의 훈련 및 추론 효율성을 증명합니다.

*   **Grinsztajn 벤치마크 (예: MiniBooNE 또는 Jannis)** 
    *   **특징:** 전통적으로 앙상블 트리 모델이 극도로 유리한 데이터셋.
    *   **목표:** 원래 TabR은 15개의 독립 모델을 학습시켜야 트리 모델을 따라잡음 (15x 연산 비용). GateR-M은 **단 1번의 학습(Single Model Retraining)**만으로 내부의 6개 패킹된 헤드를 통해 TabR 15 앙상블 스코어에 필적하거나 XGBoost를 꺾음을 증명.
    *   **검증 지표:** 훈련 시간(Training Time) 및 파라미터 수(Parameter Count)를 비교하는 Table 작성.

---

## 👨‍🏫 현재 Action Plan 상태 (진행 중)
지도교수님의 지침에 따라 다음 사항이 완벽히 세팅되어 가동 중입니다:
1. [x] **California Housing (CA)** 데이터셋 세팅 완료.
2. [x] `d_embedding` 서치 범위 `[96, 384]`, 이웃 수 `m = 96` 통제 변인 고정 완료.
3. [x] 현재 백그라운드 튜닝 중이며, SOTA(`0.400`) 타겟팅 중.
