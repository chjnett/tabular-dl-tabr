# 🧠 GateR-M (Gated Retrieval-Multilayer) 아키텍처 딥다이브

본 문서는 TabR(Gorishniy et al., 2023)의 한계점을 극복하기 위해 설계된 **GateR-M (Gated Retrieval-Multilayer)** 아키텍처의 핵심 구조와 작동 원리를 키워드 중심으로 상세히 분석한 기술 문서입니다.

---

## 🏛️ 아키텍처 오버뷰 (High-Level Flow)

GateR-M은 크게 4가지 컴포넌트의 파이프라인으로 작동합니다.

1. **입력 및 분할 (Input & Split):** 수치형 및 범주형 피처가 입력되면, 피처 단위의 독립된 임베딩 텐서(3D)로 변환됩니다.
2. **이웃 검색 (Context Retrieval):** FAISS 또는 내장 KNN 모듈을 통해 입력 샘플(Anchor)과 가장 유사한 이웃 `m`개(예: 96개)를 추출합니다.
3. **심층 검색-증강 처리 (Deep Retrieval-Augmented Layers):** N개의 누적된(Stacked) 레이어를 통과하며 Feature-Wise Attention을 통해 이웃의 정보를 앵커에 정교하게 주입합니다.
4. **고효율 앙상블 예측 (Packed Ensemble Head):** 단일 모델 내에 압축된 여러 개의 예측 헤드가 다수결(또는 평균) 예측을 수행하여 최종 값을 출력합니다.

---

## 🔑 키워드 기반 아키텍처 핵심 설계 (Core Innovations)

### 1. `Feature-Wise Projection` (피처 단위 독립 투영)
> **문제점:** 기존 TabR은 모든 피처를 하나의 긴 1D 벡터로 뭉쳐서(Flatten) 유사도를 계산하므로, 특정 피처에 섞인 노이즈가 전체 유사도 계산을 오염시킵니다.
> **해결책 (GateR-M):**
*   **작동 방식:** 텐서의 형태를 `[Batch, Features, Embed_Dim]` 형태의 **3D 텐서로 엄격하게 유지**합니다. Query, Key, Value 텐서를 투영(Projection)할 때 피처 간에 가중치를 공유하지 않고(`share_weights=False`), `torch.einsum`을 사용하여 각 피처별로 고유한 공간에 매핑합니다.
*   **효과:** "위도/경도"와 "방 개수"가 서로의 연산에 간섭하지 않으며, 특정 피처가 가진 정보량(Signal)을 독립적으로 극대화할 수 있습니다.

### 2. `Softmax Cutoff (3D Attention)` (노이즈 차단 메커니즘)
> **문제점:** 피처가 100개가 넘어가는 정형 데이터에서는 무의미한 노이즈 피처(Dummy Features)가 이웃 검색의 신뢰도를 떨어뜨립니다.
> **해결책 (GateR-M):**
*   **작동 방식:** 피처별로 독립 연산된 Query와 Key의 내적(Dot Product) 후, 피처 차원(Feature Dimension)에 대해 독립적인 Softmax 마스킹을 수행합니다. 
*   **효과:** 딥러닝이 스스로 학습 과정에서 "이 피처는 타겟 예측에 쓸모가 없다"고 판단하면, 해당 피처의 Attention 가중치를 **Softmax를 통해 0에 가깝게 차단(Cutoff)** 해버립니다. 피처가 100개, 200개로 늘어나도 성능이 강건하게(Robust) 유지되는 비결입니다.

### 3. `Stacked GateR Retrieval` (다단 심층 추상화)
> **문제점:** 기존 TabR은 이웃 데이터를 가져오는(Retrieval) 과정이 단 1번(Shallow)에 그칩니다.
> **해결책 (GateR-M):**
*   **작동 방식:** `n_layers`(예: 3~4층) 하이퍼파라미터를 통해 이웃 정보를 결합하는 어텐션 블록을 **Pre-LayerNorm 기반의 잔차 연결(Residual Connection)로 여러 겹 쌓습니다.**
*   **효과:** 첫 번째 레이어에서 얕은 표면적 유사성을 파악하고, 두 번째, 세 번째 레이어로 갈수록 이웃들 간의 고차원적(High-order) 비선형 관계를 추출하여 앵커 샘플을 심층적으로 보정(Augment)합니다. 

### 4. `Parameter-Efficient Packed Ensemble` (파라미터 효율적 패킹 앙상블)
> **문제점:** 정형 데이터 최강자인 XGBoost 등 트리 앙상블을 딥러닝이 이기기 위해, 기존 TabR은 15개의 독립된 딥러닝 모델을 처음부터 따로 학습시키는 무식하고 극도로 비효율적인 방식(15x 연산 시간)을 택했습니다.
> **해결책 (GateR-M):**
*   **작동 방식:** BatchEnsemble(Wen et al., 2020)의 개념을 차용하여, 단일 모델의 가중치 행렬 $W$에 대해 앙상블 개수(`n_ensembles=6`)만큼의 **Rank-1 스케일링 벡터($r_i, s_i$)**만을 추가로 패킹(Packing)합니다.
    $$ W_i = W \odot (r_i \otimes s_i) $$
*   **효과:** **단 1번의 훈련(Single Model Retraining)**만으로 내부적으로 6~8개의 다르게 교란된(Perturbed) 앙상블 브랜치가 생성됩니다. 파라미터 증가는 1% 미만이지만, 트리 모델에 필적하는 분산(Variance) 감소 및 안정적인 SOTA 성능 도출이 가능합니다.

---

## 🛠️ 하이퍼파라미터 설계 의도 (Optuna Search Space)

GateR-M의 거대한 표현력을 통제하기 위해 Optuna 튜닝에는 다음과 같은 철학이 반영되어 있습니다.

*   **`d_embedding (>= 96)`:** 최소 96차원 이상을 보장하여 피처별 투영과 3D 어텐션의 표현력이 훼손되지 않도록 병목(Bottleneck)을 방지합니다.
*   **`context_dropout` & `weight_decay`:** 파라미터가 비약적으로 증가한 상태이므로, 과적합(Overfitting)을 막기 위해 Context 노드들을 무작위로 끊어버리고 가중치를 규제하는 황금 비율을 탐색합니다.
