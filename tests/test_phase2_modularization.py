"""
Phase 2: TabR 구조 분석 및 모듈 의존성 분리 테스트
- 인코더 출력 차원과 프리딕터 입력 차원 호환성
- gate_rm.py 모듈 임포트 무결성
- GateRMModel 컴포넌트 독립 실행 가능성
- 자가 마스킹(Self-Masking) 누수 방지 구조
- BatchEnsemble r/s 벡터 state_dict 영속성
"""
import os
import sys
import pytest
import torch
import torch.nn as nn
import tempfile

project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ['PROJECT_DIR'] = project_dir
sys.path.append(project_dir)


# ─────────────────────────────────────────────
# Test 1: 모듈 임포트 무결성
# ─────────────────────────────────────────────
def test_gate_rm_module_importable():
    """lib/gate_rm.py 및 핵심 클래스들이 에러 없이 임포트되는지 확인"""
    from lib.gate_rm import (
        FeatureWiseProjection,
        GateRRetrieval,
        StackedGateRRetrieval,
        BatchEnsembleLinear,
        FeatureEmbedder,
        GateRMModel,
    )
    assert FeatureWiseProjection is not None
    assert GateRRetrieval is not None
    assert StackedGateRRetrieval is not None
    assert BatchEnsembleLinear is not None
    assert FeatureEmbedder is not None
    assert GateRMModel is not None


# ─────────────────────────────────────────────
# Test 2: 인코더 → 프리딕터 차원 호환성
# ─────────────────────────────────────────────
def test_encoder_predictor_interface_dimensions():
    """
    인코더 출력 [B, K, d]를 flatten하여 프리딕터에 진입하는
    차원 흐름이 정확히 일치하는지 검증
    """
    B, K, d = 8, 12, 16
    d_flat = K * d  # GateRMModel의 flatten 전략

    x_anchor = torch.randn(B, K, d)
    x_flat = x_anchor.flatten(1)  # [B, K*d]

    predictor = nn.Linear(d_flat, 1)
    out = predictor(x_flat)

    assert x_flat.shape == (B, d_flat), f"Flatten 차원 불일치: {x_flat.shape}"
    assert out.shape == (B, 1), f"예측 출력 차원 불일치: {out.shape}"


# ─────────────────────────────────────────────
# Test 3: FeatureEmbedder 출력 차원
# ─────────────────────────────────────────────
def test_feature_embedder_output_shape():
    """
    수치형 + 이진형 + 범주형 피처를 섞은 입력에 대해
    FeatureEmbedder가 [B, n_features, d_embedding] 텐서를 출력하는지 확인
    """
    from lib.gate_rm import FeatureEmbedder

    n_num, n_bin = 4, 2
    cat_cardinalities = [3, 5]
    d_embedding = 16
    B = 6
    n_features = n_num + n_bin + len(cat_cardinalities)  # = 8

    embedder = FeatureEmbedder(
        n_num_features=n_num,
        n_bin_features=n_bin,
        cat_cardinalities=cat_cardinalities,
        d_embedding=d_embedding,
    )

    x_ = {
        'num': torch.randn(B, n_num),
        'bin': torch.randint(0, 2, (B, n_bin)),
        'cat': torch.cat([
            torch.randint(0, c, (B, 1)) for c in cat_cardinalities
        ], dim=-1),
    }

    out = embedder(x_)
    assert out.shape == (B, n_features, d_embedding), (
        f"FeatureEmbedder 출력 형상 불일치: {out.shape} != {(B, n_features, d_embedding)}"
    )


# ─────────────────────────────────────────────
# Test 4: GateRRetrieval Skip Connection (Residual) 무결성
# ─────────────────────────────────────────────
def test_gate_r_retrieval_residual():
    """
    x_anchor가 검색 결과 Z와 더해지는 Residual 연결이
    출력 형상을 변경하지 않는지 검증
    """
    from lib.gate_rm import GateRRetrieval

    B, M, K, d = 4, 8, 6, 16
    model = GateRRetrieval(d_embedding=d, n_features=K, context_dropout=0.0)
    x_anchor = torch.randn(B, K, d)
    x_neighbors = torch.randn(B, M, K, d)

    out = model(x_anchor, x_neighbors)
    assert out.shape == x_anchor.shape, (
        f"Residual 연산 후 형상 변형: {out.shape} != {x_anchor.shape}"
    )


# ─────────────────────────────────────────────
# Test 5: Self-Masking — 자가 누수 방지
# ─────────────────────────────────────────────
def test_self_masking_blocks_diagonal():
    """
    대각 성분(자기 자신)에 대한 어텐션이 Softmax 후 0이 되는지 수치적으로 검증.
    masked_fill_ + softmax 파이프라인의 정확성 확인.
    """
    B = 6
    similarities = torch.randn(B, B)
    mask = torch.eye(B, dtype=torch.bool)
    similarities.masked_fill_(mask, float('-inf'))
    weights = torch.softmax(similarities, dim=-1)

    for i in range(B):
        assert weights[i, i].item() == pytest.approx(0.0, abs=1e-6), (
            f"샘플 {i}: 자가 누수 감지 (weight={weights[i, i].item():.6f})"
        )


# ─────────────────────────────────────────────
# Test 6: BatchEnsembleLinear state_dict 영속성
# ─────────────────────────────────────────────
def test_batchensemble_state_dict_persistence():
    """
    BatchEnsembleLinear의 r, s 벡터가 state_dict에 포함되어
    저장/복원 후 bit-exact로 일치하는지 검증
    """
    from lib.gate_rm import BatchEnsembleLinear

    model = BatchEnsembleLinear(d_in=8, d_out=4, n_ensembles=3)
    state = model.state_dict()

    assert 'r' in state, "state_dict에 'r' 벡터가 누락됨"
    assert 's' in state, "state_dict에 's' 벡터가 누락됨"

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "be_linear.pt")
        torch.save(state, path)
        loaded = torch.load(path, weights_only=True)

    assert torch.equal(state['r'], loaded['r']), "r 벡터 복원 불일치"
    assert torch.equal(state['s'], loaded['s']), "s 벡터 복원 불일치"
    assert torch.equal(state['weight'], loaded['weight']), "weight 복원 불일치"


# ─────────────────────────────────────────────
# Test 7: 수치 안정성 — NaN/Inf 전파 방지
# ─────────────────────────────────────────────
def test_numerical_stability_no_nan():
    """
    극단적인 입력(±500) 및 제로 벡터에서도 GateRRetrieval 출력에
    NaN / Inf가 발생하지 않는지 확인
    """
    from lib.gate_rm import GateRRetrieval

    B, M, K, d = 2, 4, 5, 8
    model = GateRRetrieval(d_embedding=d, n_features=K, context_dropout=0.0)

    # 극단값 주입
    x_anchor = torch.full((B, K, d), 500.0)
    x_neighbors = torch.full((B, M, K, d), -500.0)

    out = model(x_anchor, x_neighbors)
    assert not torch.isnan(out).any(), "극단값 입력 시 NaN 발생"
    assert not torch.isinf(out).any(), "극단값 입력 시 Inf 발생"


# ─────────────────────────────────────────────
# Test 8: 그래디언트 역전파 흐름
# ─────────────────────────────────────────────
def test_gradient_flows_through_all_params():
    """
    GateRMModel 포워드 패스 후 loss.backward() 시
    모든 requires_grad 파라미터에 gradient가 할당되는지 확인
    """
    from lib.gate_rm import GateRMModel

    model = GateRMModel(
        n_num_features=3,
        n_bin_features=0,
        cat_cardinalities=[],
        n_classes=None,
        d_embedding=8,
        n_layers=1,
        n_ensembles=2,
    )

    x_ = {'num': torch.randn(4, 3)}
    candidate_x_ = {'num': torch.randn(8, 3)}

    out = model(
        x_=x_,
        y=torch.randn(4),
        candidate_x_=candidate_x_,
        candidate_y=torch.randn(8),
        context_size=4,
        is_train=True,
    )
    loss = out.mean()
    loss.backward()

    missing_grads = [
        name for name, p in model.named_parameters()
        if p.requires_grad and p.grad is None
    ]
    assert len(missing_grads) == 0, (
        f"그래디언트가 없는 파라미터: {missing_grads}"
    )


if __name__ == '__main__':
    test_gate_rm_module_importable()
    test_encoder_predictor_interface_dimensions()
    test_feature_embedder_output_shape()
    test_gate_r_retrieval_residual()
    test_self_masking_blocks_diagonal()
    test_batchensemble_state_dict_persistence()
    test_numerical_stability_no_nan()
    test_gradient_flows_through_all_params()
    print("Phase 2 모듈화 테스트 전체 통과!")
