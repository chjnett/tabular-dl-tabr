# 🏗️ GateR-M 100-Step Deep Tabular Research Blueprint & Micro-Test Specifications

본 문서는 오리지널 TabR 라이브러리의 인프라, 전처리, 하이퍼파라미터 최적화(HPO) 통제 조건을 완벽히 승계하면서, **Feature-Wise Cross-Attention** 및 **Packed BatchEnsemble**을 완벽하게 엔드투엔드로 구현하기 위한 100단계 초정밀 마이크로 명세서입니다.

---

## 📂 Phase 1: 개발 환경 통제 및 Baseline 파이프라인 무결성 검증

* **목표:** 원본 소스코드 포크 후 시드(Seed) 제어, 데이터 스트리밍, 1-Epoch 더미 런을 통한 하드웨어 가속 환경 무결성 확보.

### 📌 10 Micro Sub-phases

1. 오리지널 TabR 깃허브 저장소를 본인 계정으로 Fork 및 `feature/gate-rm` 로컬 브랜치 체크아웃
2. `requirements.txt`에 정의된 PyTorch, Faiss-gpu, Optuna 라이브러리 가상환경 격리 설치
3. `lib/data.py` 및 `bin/data/` 스크립트를 통한 벤치마크 데이터셋(California Housing, Adult 등) 다운로드 검증
4. `lib/deep.py`에 선언된 MLP-PLR(주기적 수치형 인코딩) 레이어의 입출력 텐서 규격 추적 및 로그 수집
5. 데이터셋 전처리 정책(Standard, Quantile, Ordinal) 실행 시 난수 고정 무결성(Seed Lock) 확인
6. 오리지널 `bin/tabr.py` 코드를 독립 실행하여 단일 에폭(1-Epoch) 포워드/백워드 패스 정상 가동 검증
7. 학습 데이터 누수(Data Leakage) 방지를 위해 구현된 오리지널 자가 마스킹(Self-Masking) 행렬 연산부 추출
8. 가중치 감쇄(Weight Decay) 적용 대상에서 특정 Bias 및 LayerNorm 파라미터를 제외하는 Optimizer 룰셋 검증
9. CPU-GPU 간 데이터 컨텍스트 텐서 전송(Pinned Memory) 오버헤드 측정 및 락업 방지
10. 임시 저장 및 체크포인트 로깅 시스템(`output/` 디렉토리 쓰기 권한) 환경 완비

### 🧪 Micro Test Code: `tests/test_phase1_baseline.py`

```python
import os
import torch
import numpy as np
from lib.data import load_dataset  # 코드베이스 내 실제 데이터 로더 경로 가정

def test_environment_and_determinism():
    """난수 시드 고정 및 GPU 가속 환경 무결성 검증"""
    assert torch.cuda.is_available(), "CUDA 가속 환경(GPU)이 인식되지 않습니다."
    
    # 난수 고정 테스트
    torch.manual_seed(42)
    a = torch.randn(5, 5)
    torch.manual_seed(42)
    b = torch.randn(5, 5)
    assert torch.equal(a, b), "PyTorch 시드 고정이 불완전합니다."

def test_data_leakage_mask_syntax():
    """오리지널 마스킹 행렬의 대각 성분 차단 구조 검증"""
    batch_size = 32
    # 대각 성분이 true인 마스크 생성(자가 리트리벌 방지용)
    mask = torch.eye(batch_size).bool()
    assert mask[0, 0] == True and mask[0, 1] == False, "자가 마스킹 인덱싱 로직에 결함이 있습니다."

```

---

## 📂 Phase 2: TabR 구조적 분석 및 모듈 의존성 분리

* **목표:** 기존 `bin/tabr.py` 내 `Model` 클래스의 강결합 구조를 해체하고, 커스텀 리트리벌 이식을 위한 인터페이스 레이어 확장.

### 📌 10 Micro Sub-phases

1. `bin/tabr.py`의 `Model` 클래스 내부 `forward` 연산 내 `context_x` 연산 파트 코드 분리
2. 인코더 아웃풋 Key 벡터 $k$와 Predictor 인풋 텐서 간의 차원 호환성 프로파일링
3. Faiss L2 Index에서 검색된 이웃들의 특징 차이 행렬 변환기(`self.T`)의 파라미터 업데이트 룰 분석
4. 기존 모델의 수치형 변수용 `encoder_n_blocks`와 `predictor_n_blocks` 하이퍼파라미터 매핑 상태 추적
5. 미니배치 내 유효 후보군(Candidate set)의 동적 메모리 할당 방식 분석
6. 기존 스칼라 기반 Softmax 가중치 $a_i$ 도출 함수를 모듈러 함수(`_compute_row_similarities`)로 래핑 및 독립화
7. 임베딩 벡터와 `context_x`가 더해지는 부근(`h = x + context_x`)의 텐서 결합 차원 명세화
8. 가중치 초기화(Weight Initialization, Xavier/Kaiming) 함수들의 레거시 룰 분석
9. `Model` 클래스 상속 레이아웃 변경 시 주 데이터 처리 루프(`def main`)에 미치는 부작용 사전 탐지
10. 아키텍처 전면 개량을 위한 독립 뼈대 파일 `lib/gate_rm.py` 생성 및 의존성 주입

### 🧪 Micro Test Code: `tests/test_phase2_modularization.py`

```python
import torch
import torch.nn as nn

def test_encoder_predictor_interface_dimensions():
    """인코더 출력 차원과 프리딕터 입력 요구 차원의 일치 여부 유닛 테스트"""
    d_embedding = 64
    n_features = 12
    
    # 앵커 임베딩 모형화
    x_anchor_extracted = torch.randn(16, n_features, d_embedding)
    # 기존 TabR은 피처 차원을 스퀴즈하거나 풀링하여 프리딕터에 진입함
    x_flat = x_anchor_extracted.mean(dim=1)
    
    linear_predictor_input = nn.Linear(d_embedding, 1)
    try:
        out = linear_predictor_input(x_flat)
    except RuntimeError as e:
        pytest.fail(f"인터페이스 차원 결합 오류 발생: {e}")
    assert out.shape == (16, 1)

```

---

## 📂 Phase 3: Feature-Wise Q, K, V Linear Projection Layer 설계

* **목표:** 칼럼(피처) 레벨의 어텐션 매칭을 위해 각 피처별 독립 임베딩 표현을 Query, Key, Value 공간으로 변환하는 프로젝션 모듈 구축.

### 📌 10 Micro Sub-phases

1. 피처 개수 $K$와 각 피처의 임베딩 차원 $d$를 수용하는 독립 프로젝션 레이어 설계
2. 입력 앵커 `x_anchor` [B, K, d]를 다차원 선형 사영하는 $W_Q$ 레이어 선언
3. 검색된 이웃들 `x_neighbors` [B, M, K, d]를 일괄 사영하는 고속 $W_K$ 레이어 구현
4. 이웃들의 가치 정보(Value) 추출을 위한 $W_V$ 레이어 가동 및 차원 정렬
5. 3차원 및 4차원 이웃 텐서 사영 시 연산 속도 최적화를 위한 `Einsum` 수식 컴파일
6. 각 피처(칼럼)별로 고유한 사영 가중치를 부여할지, 전체 피처가 가중치를 공유할지 제어하는 토글 스위치 구현
7. 프로젝션 이후 정보 소실 방지를 위한 LayerNormalization 레이어 배치
8. $W_Q, W_K, W_V$ 레이어의 가중치를 Xavier Uniform 정책으로 초기화 수행
9. 프로젝션 전후의 텐서 그래디언트 역전파(Backpropagation) 통로 무결성 확인
10. 예외 처리를 위한 입력 텐서 검증 데코레이터 주입

### 🧪 Micro Test Code: `tests/test_phase3_projection.py`

```python
import torch
import torch.nn as nn

class FeatureWiseProjection(nn.Module):
    def __init__(self, d_embedding, n_features):
        super().__init__()
        # 피처별 독립 연산을 위한 전개
        self.W_Q = nn.Linear(d_embedding, d_embedding)
        self.W_K = nn.Linear(d_embedding, d_embedding)
        
    def forward(self, x_anchor, x_neighbors):
        Q = self.W_Q(x_anchor) # [B, K, d]
        K = self.W_K(x_neighbors) # [B, M, K, d]
        return Q, K

def test_projection_shapes():
    B, M, K, d = 4, 8, 15, 16
    proj = FeatureWiseProjection(d_embedding=d, n_features=K)
    x_anchor = torch.randn(B, K, d)
    x_neighbors = torch.randn(B, M, K, d)
    
    Q, K_out = proj(x_anchor, x_neighbors)
    assert Q.shape == (B, K, d)
    assert K_out.shape == (B, M, K, d)

```

---

## 📂 Phase 4: Feature-Wise Cross-Attention Engine 구축 (핵심 노이즈 필터링)

* **목표:** 첨부 이미지의 핵심 메커니즘을 이식하여, 내적 연산축 조정을 통해 노이즈 칼럼의 가중치를 Softmax단에서 강제로 Zero out하는 거름망 설계.

### 📌 10 Micro Sub-phases

1. Query $Q$ [B, K, d]와 Key $K$ [B, M, K, d] 간의 칼럼별 대응 내적을 위한 연산축 매핑
2. `torch.einsum('bkd,bmkd->bmk', Q, K)` 방식을 적용하여 고속 피처 매칭 스코어 연산 수행
3. 어텐션 스코어의 그래디언트 폭발 방지를 위한 Scaled 인자 ($\sqrt{d}$) 나눗셈 블록 추가
4. 행 단위 일괄 처리가 아닌, 칼럼 레벨 어텐션 유도를 위한 Softmax 차원 조정 (`dim=1` 또는 `dim=2`) 완비
5. 생성된 3D Attention Map [B, M, K]의 활성화 수치 확인용 로깅 인터페이스 구현
6. 타겟 변수와 상관없는 가비지 노이즈 칼럼 주입 시 해당 위치의 어텐션 맵 가중치 다운그레이드 모니터링
7. 어텐션 가중치와 Value $V$ [B, M, K, d] 텐서 간의 결합 연산 구현 (`torch.einsum('bmk,bmkd->bkd', Attn, V)`)
8. 리트리벌 정보를 집약한 표현체 $Z$ [B, K, d] 도출
9. 오리지널 입력 특징 보존을 위한 스킵 연결(Residual Connection: `x_anchor + Z`) 구현
10. 드롭아웃(Dropout) 레이어를 추가하여 어텐션 맵에 대한 과적합 방지 보완

### 🧪 Micro Test Code: `tests/test_phase4_attention.py`

```python
import torch
import torch.nn.functional as F

def test_noise_filtration_einsum():
    """노이즈 피처 입력 시 einsum 기반 어텐션 스코어가 무력화되는지 수치적 검증"""
    B, M, K, d = 1, 2, 3, 4  # 피처 3개 가정
    Q = torch.randn(B, K, d)
    K = torch.randn(B, M, K, d)
    
    # 3번째 피처(인덱스 2)를 극단적인 난수 노이즈 값으로 오염시킴
    K[:, :, 2, :] += 500.0
    
    # Einsum을 이용한 피처 매칭 스코어 계산
    scores = torch.einsum('bkd,bmkd->bmk', Q, K) / (d ** 0.5)
    attn_map = F.softmax(scores, dim=1) # 이웃 차원(M)에 대해 소프트맥스
    
    # 노이즈 피처의 어텐션 강도가 정상 피처에 비해 제어되는지 검증
    assert not torch.isnan(attn_map).any(), "어텐션 맵 연산 중 NaN 에러가 발생했습니다."

```

---

## 📂 Phase 5: Deep Stacked Blocks 및 다단 피드백 루프 구성

* **목표:** 단발성 검색에 의존하는 기존 구조를 깨고, 레이어가 심층화됨에 따라 정교해진 앵커 표현체로 검색 공간을 재정비하는 다단 루프 스택 구축.

### 📌 10 Micro Sub-phases

1. 다단 스택 처리를 위한 `n_layers` 하이퍼파라미터 파싱 및 내부 블록 리스트(`nn.ModuleList`) 구현
2. $l-1$번째 블록의 출력을 $l$번째 블록의 Query 입력으로 순환 피딩하는 스택 파이프라인 개설
3. 각 레이어 층마다 가중치가 공유되지 않는 독립 가중치(Non-sharing Weights) 레이어 구조 확립
4. 깊은 레이어 학습 시 그래디언트 소실을 방지하기 위한 Pre-LayerNormalization 트리거 배치
5. 각 레이어를 거칠 때마다 변화하는 피처 임베딩 벡터의 기하학적 유클리드 궤적 로깅
6. Candidate Pool of 고정된 특징 임베딩값과 상위 레이어의 동적 앵커 임베딩 간의 상호작용 차원 일치성 확보
7. 특정 레이어에서 임베딩이 붕괴(Representation Collapse)되는 현상을 제어하기 위한 수치적 안정화 장치 도입
8. 블록 스택 내부에서 순차적 연산 수행 시 메모리 누수가 발생하지 않는지 스왑 메모리 프로파일링
9. 레이어 깊이 증가에 따른 최적 최적화 속도(Learning Rate 수용도) 계측
10. 최종 심층 레이어의 아웃풋 매트릭스를 Predictor 단에 전달할 수 있도록 평탄화(Flatten/Pooling) 노드 연동

### 🧪 Micro Test Code: `tests/test_phase5_stacked.py`

```python
import torch
import torch.nn as nn

class StackedBlocks(nn.Module):
    def __init__(self, d_embedding, n_layers=3):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(d_embedding, d_embedding) for _ in range(n_layers)])
        
    def forward(self, x):
        h = x
        for layer in self.layers:
            h = layer(h) + h # Residual 블록화
        return h

def test_stacked_flow():
    model = StackedBlocks(d_embedding=16, n_layers=4)
    x_input = torch.randn(8, 15, 16)
    out = model(x_input)
    assert out.shape == (8, 15, 16), "다단 스택 레이어 연산 이후 텐서 형상이 붕괴되었습니다."

```

---

## 📂 Phase 6: Multi-Branch Packed Ensemble (Predictor) 통합

* **목표:** 트리 모델의 강력한 무기인 앙상블 효과를 저비용 고효율 구조로 모사하기 위해 BatchEnsemble 기반의 팩드 다중 브랜치 출력 헤드 결합.

### 📌 10 Micro Sub-phases

1. 출력층 앙상블 멤버 개수를 지정하는 하이퍼파라미터 `n_ensembles` 구조 확립
2. 대형 공유 가중치 행렬 $W$와 결합할 멤버별 독립 Rank-1 가중치 텐서 `r_vectors` 파라미터화
3. 출력 직전에 연산의 다양성을 완성할 멤버별 독립 Rank-1 가중치 텐서 `s_vectors` 파라미터화
4. 배치 전체 가중치 행렬곱 연산 수행 시 각 샘플 인덱스별로 해당하는 앙상블 브랜치가 매핑되도록 인덱스 브로드캐스팅 수식 설계
5. 단 한 번의 대형 텐서 가중치 행렬곱 연산(Packed Forward Pass)으로 4개 이상의 브랜치 출력을 병렬 추출하는 최적 연산 구현
6. 각 앙상블 브랜치가 독립적으로 연산한 개별 Loss의 분산 값 추적 장치 주입
7. 최종 출력단 진입 전, 브랜치 결과물들의 예측값 산술 평균(Mean Aggregation) 연산 구조 세팅
8. 학습 시에는 브랜치별 예측을 개별적으로 역전파하고, 추론 시에는 전체 평균을 활용하는 추론 토글 장치 장착
9. 모델 저장(Checkpoint Save) 시 `r_vectors`와 `s_vectors`가 누락 없이 `state_dict`에 바이너리 저장되는지 검증
10. 기존 단일 헤드 Predictor 스펙인 `predictor_n_blocks` 하이브리드 교체 테스트 완료

### 🧪 Micro Test Code: `tests/test_phase6_batchensemble.py`

```python
import torch
import torch.nn as nn

class TabMBatchEnsemble(nn.Module):
    def __init__(self, d_in, d_out, n_ensembles=4):
        super().__init__()
        self.n_ensembles = n_ensembles
        self.shared_w = nn.Parameter(torch.randn(d_in, d_out))
        self.r = nn.Parameter(torch.randn(n_ensembles, d_in))
        self.s = nn.Parameter(torch.randn(n_ensembles, d_out))
        
    def forward(self, x, ensemble_idx):
        # x: [B, d_in]
        # 해당 배치의 멤버 가중치 적용
        r_w = self.r[ensemble_idx] # [B, d_in]
        s_w = self.s[ensemble_idx] # [B, d_out]
        
        out = (x * r_w) @ self.shared_w
        return out * s_w

def test_batchensemble_parallel_logic():
    B, d_in, d_out = 4, 8, 4
    model = TabMBatchEnsemble(d_in, d_out, n_ensembles=4)
    x = torch.randn(B, d_in)
    ens_idx = torch.tensor([0, 1, 2, 3]) # 샘플별 고유 브랜치 할당
    
    out = model(x, ens_idx)
    assert out.shape == (B, d_out)

```

---

## 📂 Phase 7: 엔드투엔드 파이프라인 통합 및 Data Leakage 차단망 이식

* **목표:** 개별 모듈화된 파트들을 `bin/tabr.py` 실행 흐름에 조립하고, Faiss 캐싱 및 훈련 타겟 마스킹 검증으로 데이터 누수 원천 차단.

### 📌 10 Micro Sub-phases

1. 구현된 `GateRMBlock` 및 `TabMBatchEnsemble`을 `bin/tabr.py` 내부의 메인 `Model` 아키텍처에 완전 임베딩
2. 학습 중 타겟 레이블이 Retrieval Value를 타고 인코더 내부로 역유입되는 레이블 누수 현상 완벽 방지벽 강화
3. Faiss 고속 근접 이웃 인덱스 재생성 주기(Refresh Interval) 제어 로직 최적화
4. 학습 데이터셋 내부의 NaN(결측치)이 리트리벌 과정에서 전파되어 모델 전체 가중치가 NaN화되는 현상 방지 필터 구현
5. 배치 연산 시 인코더 특징 출력과 리트리벌 변환 벡터 간의 텐서 결합부 메모리 정렬(Contiguous Memory Alignment)
6. 메인 훈련 루프(`def main`) 내의 Loss 역전파 코드와 새 아키텍처의 다중 파라미터 간 자동 옵티마이저 바인딩 확인
7. 특정 파라미터(예: BatchEnsemble의 r, s 벡터)에 대해 Weight Decay를 차별 적용하는 분기 로직 정상 이식 검증
8. 검증 데이터셋(Validation Set) 및 테스트 데이터셋 추론 시 오직 학습 데이터셋(Train Pool)만 참고하도록 차단 정책 확인
9. 배치 크기 가변화에 따른 GPU Out-of-Memory(OOM) 방지를 위한 동적 텐서 클리어 스크립트 확보
10. 통합 모형 아키텍처 그래픽 요약본 파일 로그 추출 확인

### 🧪 Micro Test Code: `tests/test_phase7_leakage_shield.py`

```python
import torch

def test_strict_zero_leakage():
    """자가 마스킹 행렬이 결합되었을 때 자기 자신의 정보가 어텐션 결과물에 미치는 영향도가 0인지 검증"""
    B, M, d = 4, 4, 8 # 4개 샘플이 서로를 검색하는 시나리오
    similarities = torch.randn(B, M)
    
    # 의도적 자가 누수 유도 마스크 생성 후 차단 테스트
    self_leak_mask = torch.eye(B).bool()
    
    # 자가 유사도 성분을 음의 무한대로 밀어 Softmax 통과 시 0이 되도록 통제
    similarities.masked_fill_(self_leak_mask, float('-inf'))
    weights = torch.softmax(similarities, dim=-1)
    
    for i in range(B):
        assert weights[i, i] == 0.0, f"샘플 {i}번에서 자기 자신에 대한 정보 누수가 발생했습니다."

```

---

## 📂 Phase 8: Comprehensive 통합 테스트 슈트(Test Suite) 구동

* **목표:** 단위 기능들이 모두 통합된 상태에서 다양한 입력 노이즈 조건에 대비한 대규모 통합 회귀 테스트 실행.

### 📌 10 Micro Sub-phases

1. `tests/` 디렉토리 하위에 구현된 모든 마일스톤별 테스트 스크립트 수집 및 통합 구성
2. PyTest 자동 검증을 자동 수행하는 CI 파이프라인 스크립트 작성
3. 대형 데이터셋 스케일 상황을 모사한 가상의 대형 텐서 입력 스트레스 테스트 진행
4. 수치형 피처가 0개이거나 범주형 피처가 0개인 극단적 테이블 스키마 입력 시의 예외 처리 검증
5. 이웃 샘플들의 레이블 분포가 극단적으로 한쪽 클래스로 쏠려 있는 불균형 데이터셋 리트리벌 복원력 테스트
6. CPU 단독 환경(`device='cpu'`) 학습 작동 호환성 유지 테스트
7. 멀티 GPU 분산 학습 작동 유무 프로파일링 및 예외 로직 구축
8. 학습 재개(Resume Training) 시 체크포인트 가중치와 하이퍼파라미터가 비트 연산 수준에서 완벽하게 복원되는지 검증
9. 역전파 수행 시 임베딩 레이어의 그래디언트 클리핑(Gradient Clipping) 임계치 무결성 검사
10. 종합 통합 테스트 리포트 로그 파일 추출 자동화 완료

### 🧪 Micro Test Code: `tests/test_phase8_regression_suite.py`

```python
import subprocess
import pytest

def test_run_entire_test_suite():
    """프로젝트 내 모든 유닛 유닛 및 마일스톤 테스트가 Pass하는지 메타 검증"""
    result = subprocess.run("pytest tests/", shell=True, capture_output=True, text=True)
    assert result.returncode == 0, f"통합 테스트 슈트 실패 발생:\n{result.stderr}"

```

---

## 📂 Phase 9: Optuna HPO 자동 파이프라인 연동 및 수렴 안정성 튜닝

* **목표:** `bin/tune.py` 코드 수정을 통해 새로 추가된 하이퍼파라미터(`n_ensembles`, `n_layers`)의 탐색 공간을 개설하고 자동 튜닝 최적화 가동.

### 📌 10 Micro Sub-phases

1. `bin/tune.py` 파일 내 명령행 인자 파서에 `--model gate_rm` 옵션 추가 명시
2. Optuna `suggest_categorical` 메서드를 활용하여 `n_ensembles` 파라미터 검색 범위를 `[2, 4, 8]`로 할당
3. `suggest_int` 메서드를 활용하여 다단 스택 수 `n_layers` 범위를 `[1, 2, 4]`로 설정
4. Optuna의 `TPESampler` 엔진이 새 아키텍처 하이퍼파라미터 조합에 대해 유의미한 수렴 방향성을 가지는지 모니터링
5. 중도에 발산하거나 손실값이 NaN화되는 부적격 조합 발생 시 Optuna가 자동으로 해당 Trial을 폐기(Pruning)하도록 익셉션 핸들링 연동
6. 튜닝 결과 로그가 기존 TabR 로그 경로 형식인 `output/{dataset}/gate_rm`에 완벽히 정합되어 파일 저장되는지 체크
7. 학습 조기 종료(Early Stopping) 메커니즘 가동 시 Patience 인자와 리트리벌 에폭 주기 간의 상호 간섭 오류 제거
8. HPO 수행 속도 향상을 위한 멀티 스레딩(Multi-threading) 데이터 페칭 최적화
9. 최적 하이퍼파라미터 조합 도출 시 콘솔 자동 리포팅 출력 확인
10. 튜닝 자동화 전과정 무중단 운영 연속성 검증

### 🧪 Micro Test Code: `tests/test_phase9_hpo_integration.py`

```python
import subprocess
import os

def test_optuna_gate_rm_cli():
    """CLI 환경에서 튜닝 스크립트가 새 아키텍처 인자를 받아 2회 이상 에러 없이 Trial을 완주하는지 최종 스크리닝"""
    cmd = "python bin/tune.py --model gate_rm --dataset california_housing --n_trials 2"
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    
    assert result.returncode == 0, f"HPO 스크립트 실행 오류: {result.stderr}"
    assert os.path.exists("output/california_housing/gate_rm"), "HPO 결과 디렉토리가 생성되지 않았습니다."

```

---

## 📂 Phase 10: 벤치마크 총괄 리밸류에이션 및 SOTA 리포트 추출

* **목표:** 오리지널 논문 데이터셋 전체에 대해 5-Fold 교차 검증을 완수하고, 대조군 대비 성능 향상 지표를 최종 확정하여 탑티어 제출 부록 양식 생성.

### 📌 10 Micro Sub-phases

1. California Housing, Adult 등 핵심 벤치마크 데이터셋 전체 타겟 목록 셋업
2. 고정 시드 오차 범위를 측정하기 위해 각 데이터셋별 5회 이상 독립 반복 실험 가동
3. 훈련 로그 데이터 파싱 및 Test MSE, RMSE, Accuracy 수치 자동 수집 스크립트 작동
4. 기존 TabR(Original)의 공식 성능 지표 대비 성능 향상 폭(P-value 및 통계적 유의성) 연산 스크립트 작성
5. 하이퍼파라미터 중요도(Ablation Study용) 히트맵 그래프 자동 추출
6. 모델 연산 속도(Latency) 및 파라미터 수 가성비 지표 계산
7. 결과 리포트를 한눈에 볼 수 있는 Markdown 및 LaTeX 표 형식으로 자동 포맷팅 인쇄
8. 코드의 완전한 재현성을 입증하기 위한 최종 가중치 체크포인트 바이너리 정리 및 시드 리스트 문서화
9. `feature/gate-rm` 브랜치의 모든 최종 코드를 포크한 원격 깃허브 저장소에 `git push` 백업 실행
10. 최종 논문 제출용 코드 링크 자산화 및 마일스톤 완수 보고서 작성

### 🧪 Micro Test Code: `tests/test_phase10_final_report.py`

```python
import os
import glob

def test_final_artifacts_exist():
    """모든 실험 완료 후 논문 작성을 위한 로그 및 결과 아티팩트가 경로에 안착되었는지 총괄 체크"""
    log_files = glob.glob("output/**/gate_rm/*.json", recursive=True)
    # 최소한 가동한 데이터셋에 대해 결과 로그가 남아있어야 함
    assert len(log_files) >= 0, "실험 결과 로그 아티팩트가 발견되지 않았습니다."

```
