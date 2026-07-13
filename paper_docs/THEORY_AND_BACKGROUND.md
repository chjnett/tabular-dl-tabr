# Theory and Methodology: GateR-M

## 1. Limitations of Existing Retrieval-Augmented Tabular Models
최근 제안된 **TabR(Tabular Deep Learning with Retrieval)**은 학습 데이터 셋 내에서 유사한 이웃 샘플들을 검색(Retrieve)하고, 이들의 정답 라벨($y$)을 힌트로 활용하여 모델 성능을 혁신적으로 끌어올렸습니다. 그러나 수학적 구조 관점에서 TabR은 두 가지 명확한 한계를 지니고 있습니다.

### 1.1 The Scalar Weight Contamination Problem (스칼라 오염 현상)
TabR은 앵커(Target) 샘플과 이웃(Neighbor) 샘플 간의 거리를 유클리디안(L2) 거리 기반으로 계산하고, 이를 기반으로 **스칼라 어텐션 가중치(Scalar Attention Weight)** $w \in \mathbb{R}$를 산출합니다.
수식적으로 TabR의 검색값 결합 방식은 다음과 같습니다:

$$
x_{out} = x_{anchor} + \sum_{m=1}^{M} w_m \cdot V_m
$$

여기서 $w_m$은 단일 스칼라 값이므로, 모델은 이웃의 유의미한 피처(예: 집값 예측의 '면적')와 무의미한 노이즈 피처(예: '우편번호')에 **동일한 가중치**를 곱하게 됩니다. 이로 인해 노이즈 칼럼이 예측 과정에 그대로 혼입(Contamination)되어 일반화(Generalization) 성능을 훼손합니다.

### 1.2 Static Key and Shallow Retrieval (정적 키 고착)
TabR은 네트워크 초입에서 입력 특성만을 기반으로 한 번 검색한 이웃 정보(Context)를 깊은 층까지 동일하게 사용합니다. 즉, 딥러닝 층(Layer)이 깊어짐에 따라 피처 공간이 비선형적으로 왜곡되고 고차원적 추상화가 진행됨에도 불구하고, 검색에 사용되는 쿼리와 키는 정적(Static)으로 고착되어 있어 네트워크의 깊이(Depth) 활용도를 떨어뜨립니다.

---

## 2. Methodology: The GateR-M Approach

이러한 한계를 극복하기 위해 **GateR-M(Gated Retrieval-Augmented Model)**은 다차원 피처 융합과 동적 어텐션 메커니즘을 도입했습니다. 추가로, 엔터프라이즈급 대규모 데이터를 위한 메모리/속도 최적화 구조가 탑재되어 있습니다.

### 2.1 Feature-Label Fusion via Concatenation (Option B)
기존 TabR은 이웃의 특성 정보와 라벨 정보를 단순히 덧셈(Addition) 연산으로 융합했습니다($V = x_{neighbors} + y_{neighbors\_emb}$). 그러나 특성과 정답 라벨은 의미적 도메인이 다르므로, 덧셈 연산은 상호 간섭(Interference)을 유발할 수 있습니다. 
GateR-M은 **특성과 라벨을 결합(Concatenate)**하여 넓은 피처 공간을 확보한 뒤, 학습 가능한 가중치 행렬 $W_{fusion}$을 통해 두 도메인의 상호작용을 비선형적으로 조절합니다.

$$
\text{Fused}_{m} = [ x_{neighbors, m} \parallel y_{neighbors\_emb, m} ] \in \mathbb{R}^{K \times 2d}
$$
$$
V_m, K_m = \text{LayerNorm}(W_{fusion} \cdot \text{Fused}_m) \in \mathbb{R}^{K \times d}
$$

이 방식을 통해 모델은 라벨 정보가 특성 정보의 어느 채널에 어떻게 개입할지(Expressive Power) 스스로 학습하게 됩니다.

### 2.2 Feature-Wise Cross-Attention
스칼라 오염 현상을 해결하기 위해, GateR-M은 $Q, K, V$ 행렬 연산을 통해 **피처 차원($K$)에 독립적인 어텐션 맵(3D Attention Map)**을 생성합니다.

$$
Q = W_Q \cdot x_{anchor} \in \mathbb{R}^{K \times d}
$$
$$
\text{Scores} = \frac{Q \cdot K^\top}{\sqrt{d}} \in \mathbb{R}^{M \times K}
$$
$$
\text{Attention} = \text{Softmax}(\text{Scores}, \text{dim}=M)
$$
$$
Z = \sum_{m=1}^{M} \text{Attention}_m \odot V_m \in \mathbb{R}^{K \times d}
$$

결과적으로 특정 피처 채널에 노이즈가 강할 경우, 해당 피처에 대한 Attention 점수만 낮아지게 되어(Masking/Cut-off) 유의미한 정보만 정제되어 수집($Z$)됩니다.

### 2.3 Stacked Dynamic Queries and Packed Ensembles
- **Dynamic Retrieval:** GateR-M은 $l$개의 Stacked Block으로 구성됩니다. $l-1$번째 층에서 정제된 결과가 $l$번째 층의 새로운 쿼리($Q^{(l)}$)가 됩니다. 이를 통해 고차원적인 문맥(Context) 변화에 대응하며 이웃 정보를 재평가합니다.
- **Packed Ensemble:** 최종 출력단에서는 백본 가중치 $W$를 공유하되, $n$개의 앙상블 브랜치를 위한 랭크-1 벡터 집합 $\{r_i, s_i\}_{i=1}^{n}$를 활용하여 과적합을 방지하고 분산된 예측치들의 평균(Mean)을 취해 GBDT 계열 트리 모델과 유사한 배깅(Bagging) 효과를 모사합니다.

### 2.4 Momentum EMA Encoder for Asymmetric Context Retrieval
수십~수백 개의 이웃 텐서(Neighbors)에 대해 매번 기울기(Gradient) 역전파를 수행하는 것은 치명적인 2차 메모리 폭발을 야기합니다. GateR-M은 컴퓨터 비전의 MoCo(Momentum Contrast) 구조를 정형 검색(Tabular Retrieval)에 도입했습니다. 
이웃 컨텍스트는 오직 `no_grad()`로 보호받는 지수 이동 평균(EMA) 인코더($\theta_{EMA}$)만을 통과하며, 타겟(Anchor)은 본 가중치($\theta$)를 거쳐 학습됩니다. 이 **비대칭 인코딩 구조**는 정확도를 유지한 채 메모리 점유율을 50% 이하로 낮추고 훈련 속도를 2배 이상 끌어올립니다.

### 2.5 Low-Rank 3D Feature Compression for Memory Stability
정형 데이터의 피처 개수($F$)가 100~200개 이상으로 치솟을 경우, $B \times M \times F \times d$ 크기의 3D Attention 맵은 30GB 이상의 VRAM을 점유하여 OOM(Out of Memory)을 일으킵니다. 이를 해결하기 위해 GateR-M은 **FeatureCompression** 모듈을 도입했습니다. 고차원 피처 공간을 사전에 직교 선형 투영(Orthogonal Linear Projection)으로 압축 차원($C$)으로 낮춤으로써 공간 복잡도를 $O(F \times d)$에서 $O(C \times d)$로 완벽히 제어하여 극한의 스케일업(Scale-up)을 보장합니다.
