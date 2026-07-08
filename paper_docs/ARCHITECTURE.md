# GateR-M: Architecture and Tensor Flow

## 1. Overview
**GateR-M (Gated Retrieval-Augmented Models for Tabular Data)**는 기존 SOTA Retrieval 모델인 **TabR**의 구조적 한계를 극복하기 위해 설계된 혁신적인 아키텍처입니다. TabR이 지닌 'Row-wise 스칼라 가중치로 인한 노이즈 칼럼 오염 현상'과 '정적 키(Static Key) 고착 문제', 그리고 '단일 헤드의 과적합 문제'를 해결하기 위해 3가지 핵심 기여도를 제안합니다:

1. **Feature-Wise Cross-Attention (특성-라벨 Concat 방식 - 옵션 B):** 단순 스칼라 어텐션이 아닌 피처(Feature) 단위의 3D Attention을 수행하며, 이때 이웃의 특성 정보와 정답(Label) 정보를 피처 차원에서 결합(Concat)하여 더욱 풍부한 검색 공간을 구축합니다.
2. **Deep Retrieval-Augmented Layers (Stacked Blocks):** 단발성 검색에 그치지 않고 여러 계층에 걸쳐 검색된 이웃 정보를 앵커(Anchor)에 동적으로 반영하는 다단(Multi-layer) 구조를 채택했습니다.
3. **Parameter-Efficient Packed Ensemble Predictor:** 최소한의 파라미터(Rank-1 가중치) 추가만으로 다수의 독립된 앙상블 브랜치를 생성해 트리 모델(GBDT) 수준의 일반화 방어력을 확보합니다.
4. **Non-Parametric Memory & Real-Time Update:** 훈련된 피처 임베딩과 라벨 임베딩을 메모리 뱅크(`memory_k`, `memory_y_emb`)에 캐싱하여 추론 병목을 제거했으며, 모델 재학습(Retraining) 없이 O(1) 연산(`append_memory`)만으로 실시간 트렌드를 예측에 반영할 수 있습니다.

## 2. Tensor Flow & Dimensionality Tracking
입력 데이터에서 최종 예측값까지의 텐서 셰이프 변화를 추적합니다.
- $B$: Batch Size (미니배치 크기)
- $M$: Context Size (검색된 이웃의 수)
- $K$: Number of Features (피처의 총 개수)
- $d$: Embedding Dimension (`d_embedding`)

### 2.1 Feature Embedding & Label Encoding
- **Input Features ($x_{anchor}$):** `[B, K, d]`
- **Neighbor Features ($x_{neighbors}$):** `[B, M, K, d]`
- **Neighbor Labels ($candidate\_y[context\_idx]$):** `[B, M]`
- **Label Encoding ($y_{neighbors\_emb}$):** `[B, M, d]`
  - *회귀(Regression) 시 `nn.Linear(1, d)`, 분류(Classification) 시 `nn.Embedding` 적용.*
  - 피처 차원(K)과 결합하기 위해 Broadcasting 되어 `[B, M, K, d]`로 취급됩니다.

### 2.2 Feature-Label Fusion (Option B: Concat 방식)
이웃의 특성과 라벨 임베딩을 단순히 더하지 않고 결합(Concat)하여 정보 손실을 최소화합니다.
- **Fusion (Concat):** `fused = torch.cat([x_neighbors, label_emb], dim=-1)`
  - 셰이프 변화: `[B, M, K, d]` + `[B, M, 1, d]`(Broadcast) $\rightarrow$ **`[B, M, K, 2d]`**
- **Linear Projection ($W_{fusion}$):** `self.label_fusion(fused)`
  - 셰이프 변화: `[B, M, K, 2d]` $\rightarrow$ **`[B, M, K, d]`**

### 2.3 Feature-Wise Cross-Attention
결합된 이웃 정보로부터 Key와 Value를, 앵커에서 Query를 생성하여 어텐션을 수행합니다.
- **Q (from $x_{anchor}$):** `[B, K, d]`
- **K, V (from $W_{fusion}$ output):** `[B, M, K, d]`
- **Scores ($Q \cdot K$):** `[B, M, K]`
  - 기존 TabR은 `[B, M]` 차원의 스칼라 점수만을 냈으나, GateR-M은 각 피처($K$)마다 독립적인 어텐션 가중치를 부여합니다.
- **Attention Map ($Softmax(Scores)$):** `[B, M, K]`
- **Values Aggregation ($Z$):** `[B, K, d]`
  - `torch.einsum('bmk,bmkd->bkd', attn, V)`

### 2.4 Stacked Retrieval & Ensemble Prediction
- **Residual Connection:** `h = x_anchor + Z` (`[B, K, d]`)
- **Flattening:** `h.flatten(1)` $\rightarrow$ `[B, K * d]`
- **Packed Ensemble Output:** `[B, n_ensembles, d_out]` (평가 시 평균을 내어 `[B, d_out]`으로 도출)

### 2.5 Non-Parametric Memory Caching (Inference Optimization)
추론(Inference) 단계에서 수만 개의 훈련 데이터를 매 뱃치마다 재인코딩하던 오버헤드를 제거합니다.
- `init_memory()`: 전체 `candidate_x_`와 `candidate_y`를 최초 1회 인코딩하여 `self.memory_k`와 `self.memory_y_emb`에 저장합니다.
- `append_memory()`: 실시간으로 새 데이터(예: 일일 인입 데이터)가 발생 시, $O(1)$ 복잡도로 인코딩한 텐서(`new_k`, `new_y_emb`)를 기존 `memory_k`, `memory_y_emb` 뒤에 `torch.cat`으로 이어붙여 즉각적인 트렌드 반영을 수행합니다.

## 3. Code Implementation Mapping
`lib/gate_rm.py` 소스 코드에 구현된 모듈과 논문의 구조를 다음과 같이 매핑할 수 있습니다.

| 논문 모듈 명명 | 소스 코드 구현체 | 설명 |
|---|---|---|
| **GateRM Backbone** | `GateRMModel` | 전체 모델의 엔트리 포인트. 데이터 인코딩, 이웃 거리 계산(FAISS/cdist), Top-K 인덱싱 및 Label Extraction(`candidate_y`)을 총괄 수행합니다. |
| **Feature Embedder** | `FeatureEmbedder` | 수치형/범주형 데이터를 일관된 차원 `d`로 투영하여 `[B, K, d]` 텐서를 생성합니다. |
| **Stacked Retrieval Blocks** | `StackedGateRRetrieval` | 다층(`n_layers`)의 검색 레이어를 구성하며, Pre-LayerNorm 방식의 Residual 연결을 제어합니다. |
| **Feature-Label Fusion & Attention** | `GateRRetrieval` | **(옵션 B 핵심 적용부)** 이웃 특성과 라벨 임베딩을 Concat 후 Linear 투영(`label_fusion`)하고, `FeatureWiseProjection`을 통해 어텐션을 수행합니다. |
| **Packed Ensemble Head** | `BatchEnsembleLinear` | 단일 Linear 가중치 $W$에 대해 $n\_ensembles$ 차원의 Rank-1 벡터($r$, $s$) 모음을 점진적으로 곱하여 연산량을 억제한 앙상블 출력을 만듭니다. |
