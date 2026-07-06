# TabR 모델 아키텍처 및 훈련 파이프라인 분석

본 문서는 공식 TabR(Tabular Deep Learning Meets Nearest Neighbors) 코드베이스의 아키텍처 및 훈련/튜닝 인터페이스 구조를 분석한 문서입니다.

---

## 1. 핵심 파일 위치 및 역할

이 프로젝트의 핵심 파일들은 다음과 같이 배치되어 있습니다:

- **`bin/tabr.py`**:
  - TabR 모델의 PyTorch 정의(`class Model`)와 단일 실험을 수행하는 메인 실행 인터페이스(`def main`)가 포함되어 있습니다.
- **`bin/tune.py`**:
  - Optuna를 기반으로 하이퍼파라미터 탐색(HPO)을 제어하는 스크립트입니다.
- **`lib/deep.py`**:
  - 모델 내부에서 사용하는 임베딩 레이어(수치형 피처를 위한 `LinearEmbeddings`, `PeriodicEmbeddings` 등, 범주형 피처를 위한 `CatEmbeddings` 등)와 최적화 도구(Optimizer 생성) 등 딥러닝 유틸리티가 정의되어 있습니다.
- **`lib/data.py`**:
  - 데이터셋 객체(`Dataset`) 정의 및 표준화/정규화 등 전처리 정책(`NumPolicy`, `CatPolicy`, `YPolicy`)을 관리합니다.

---

## 2. TabR 모델 아키텍처 분석 (`bin/tabr.py`의 `class Model`)

TabR 아키텍처는 크게 세 가지 모듈로 나뉩니다:
1. **Encoder (E)**
2. **Retrieval Module / Interaction (R)**
3. **Predictor (P)**

### 2.1. Encoder (E)
입력 피처들을 고차원 표상(`d_main`)으로 매핑하는 단계입니다.
- **피처 임베딩**: 수치형 피처(`X_num`), 범주형 피처(`X_cat`), 이진 피처(`X_bin`)를 각각 인코딩한 후 하나로 접합(concatenate)합니다. 수치형 피처의 경우 `num_embeddings` 옵션에 따라 원본 유지 혹은 선형/주기 임베딩을 거칩니다.
- **인코더 MLP 블록**: 접합된 피처 벡터는 선형 레이어를 거친 후 `encoder_n_blocks`개의 Residual 블록(MLP)을 통과하여 최종 특징 벡터 $x$를 형성합니다.
- **Key 생성**: 특징 벡터 $x$는 선형 레이어 `self.K`를 통과하여 검색용 Key 벡터 $k$로 변환됩니다.

### 2.2. Retrieval Module (R)
데이터셋 내의 후보군(Candidate set)으로부터 유사한 이웃들을 찾고 정보를 섞는 핵심 모듈입니다.
- **FAISS 검색**: PyTorch 상에서 FAISS GPU/CPU 인덱스(`faiss.IndexFlatL2`)를 이용해 현재 배치의 Key 벡터 $k$와 후보 Key 벡터 `candidate_k` 사이의 L2 거리를 기준으로 인접한 `context_size`개의 이웃을 빠르게 검색합니다.
- **누수 방지 (Leakage Prevention)**: 학습 시기(`is_train=True`)에는 자기 자신을 이웃 검색에서 배제하기 위해 후보군 최상단에 현재 배치($k$, $y$)를 배치하고 검색 후 자기 자신에 해당하는 거리를 무한대(`torch.inf`)로 마스킹합니다.
- **유사도(Similarities) & 가중치(Softmax)**:
  - 타겟 오브젝트 $k$와 컨텍스트(이웃) 오브젝트들 $context\_k$ 간의 정확한 Pairwise L2 거리를 역산하여 유사도를 계산합니다.
  - 이 유사도 벡터에 Softmax를 적용하여 각 이웃에 대한 가중치(attention weights) `probs`를 얻고, Dropout을 적용합니다.
- **값(Values) 생성**:
  - 각 이웃의 타겟 값(라벨 $y$)을 `label_encoder`를 통해 벡터로 매핑합니다.
  - 타겟 $k$와 이웃 $context\_k$의 차이 벡터를 비선형 네트워크 `self.T`에 통과시켜 잔차 성분을 생성합니다.
  - 이 둘을 더해 각 이웃의 최종 표현 값 `values = context_y_emb + self.T(k - context_k)`을 계산합니다.
- **정보 통합 (Context Aggregation)**:
  - 가중치 `probs`와 `values`를 행렬곱하여 이웃 정보를 반영한 컨텍스트 벡터 `context_x`를 구하고, 이를 원본 인코더 출력 $x$에 잔차 형식으로 더해줍니다 (`x = x + context_x`).

### 2.3. Predictor (P)
통합된 벡터 $x$로부터 최종 예측값(Logits 또는 Regression output)을 도출합니다.
- **예측 MLP 블록**: `predictor_n_blocks`개의 Residual 블록을 추가로 거치며 학습 강도를 높입니다.
- **출력 헤드**: 최종 정규화 및 활성화 함수를 통과한 후 Linear Head를 거쳐 문제 형태(분류/회귀)에 맞는 최종 아웃풋 차원으로 사영합니다.

---

## 3. 훈련 및 튜닝 인터페이스 분석

### 3.1. 단일 실행 흐름 (`bin/tabr.py` -> `main`)
- **Dataset 로드**: `lib.build_dataset`을 통해 학습용, 검증용, 평가용 데이터를 로드합니다.
- **Model 초기화 및 GPU 전송**: `Model` 객체를 생성하고 최적 장치(GPU)로 보냅니다.
- **Optimizer 설정**: 특정 레이어(예: `label_encoder`) 및 편향(bias) 등에 대해 Weight Decay를 적용하지 않도록 룰링된 `zero_wd_condition`을 적용해 Optimizer를 빌드합니다.
- **에폭 단위 훈련 루프**:
  - `lib.make_random_batches`를 사용해 배치를 무작위로 추출합니다.
  - 후보군(`candidate_x`, `candidate_y`)에서 현재 훈련 배치에 속한 샘플을 제거(autograd 연산 누수 및 치팅 방지)한 후 모델의 `forward`로 전달합니다.
  - 매 에폭 종료 후 `evaluate` 메서드로 Validation 및 Test 성능 평가를 수행하며, 최고 Validation 점수를 갱신할 때마다 최적 체크포인트를 저장합니다.

### 3.2. 하이퍼파라미터 튜닝 (`bin/tune.py`)
- Optuna의 `TPESampler`를 기반으로 정의된 파라미터 검색 공간(`space`)에서 값을 샘플링합니다.
- 목적 함수(`objective`)는 내부적으로 `bin/tabr.py`의 `main` 함수를 호출하여 검증 데이터셋의 최고 스코어(Score)를 타겟으로 최대화(maximize)합니다.
- 튜닝 진행 중 중단되더라도 재시작할 수 있도록 중간 상태(Study, Trial Reports 등)를 체크포인트로 저장합니다.
