"""
Phase 9: Optuna HPO 자동 파이프라인 통합 테스트
- Optuna 임포트 및 Trial 샘플링 가능 여부
- gate_rm 탐색 공간 파라미터 정의 유효성
- 2회 Trial 완주 시뮬레이션
- GateRMModel이 다양한 HPO 샘플 조합에서 에러 없이 구동되는지 확인
- Pruning(조기 종료) 메커니즘 작동
"""
import os
import sys
import pytest
import torch
import optuna

project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ['PROJECT_DIR'] = project_dir
sys.path.append(project_dir)

from lib.gate_rm import GateRMModel

# Optuna 로그 억제
optuna.logging.set_verbosity(optuna.logging.WARNING)


# ─────────────────────────────────────────────
# 공통: gate_rm용 HPO 탐색 공간 정의
# ─────────────────────────────────────────────
GATE_RM_SEARCH_SPACE = {
    "d_embedding": ("int", 8, 32),
    "n_layers":    ("int", 1, 4),
    "n_ensembles": ("int", 2, 8),
    "context_dropout": ("float", 0.0, 0.5),
    "lr": ("loguniform", 1e-5, 1e-2),
}


def sample_gate_rm_params(trial: optuna.Trial) -> dict:
    """Optuna Trial에서 gate_rm 하이퍼파라미터를 샘플링"""
    return {
        "d_embedding":     trial.suggest_int("d_embedding", 8, 32, step=4),
        "n_layers":        trial.suggest_int("n_layers", 1, 4),
        "n_ensembles":     trial.suggest_int("n_ensembles", 2, 8, step=2),
        "context_dropout": trial.suggest_float("context_dropout", 0.0, 0.5),
        "lr":              trial.suggest_float("lr", 1e-5, 1e-2, log=True),
    }


# ─────────────────────────────────────────────
# Test 1: Optuna 임포트 및 기본 스터디 생성
# ─────────────────────────────────────────────
def test_optuna_importable_and_study_creation():
    """Optuna 패키지가 정상 임포트되고 Study가 생성되는지 확인"""
    study = optuna.create_study(direction="minimize")
    assert study is not None
    assert study.direction == optuna.study.StudyDirection.MINIMIZE


# ─────────────────────────────────────────────
# Test 2: 탐색 공간 파라미터 샘플링 유효성
# ─────────────────────────────────────────────
def test_search_space_sampling_valid():
    """
    gate_rm 탐색 공간에서 샘플링한 파라미터가
    각 범위 안에 속하는지 10회 반복 검증
    """
    study = optuna.create_study(direction="minimize")
    
    sampled_params = []
    for _ in range(10):
        trial = study.ask()
        params = sample_gate_rm_params(trial)
        sampled_params.append(params)
        study.tell(trial, 0.5)  # 더미 목적함수 값

    for params in sampled_params:
        assert params["d_embedding"] in range(8, 33, 4), (
            f"d_embedding 범위 초과: {params['d_embedding']}"
        )
        assert 1 <= params["n_layers"] <= 4, (
            f"n_layers 범위 초과: {params['n_layers']}"
        )
        assert 2 <= params["n_ensembles"] <= 8, (
            f"n_ensembles 범위 초과: {params['n_ensembles']}"
        )
        assert 0.0 <= params["context_dropout"] <= 0.5, (
            f"context_dropout 범위 초과: {params['context_dropout']}"
        )
        assert 1e-5 <= params["lr"] <= 1e-2, (
            f"lr 범위 초과: {params['lr']}"
        )


# ─────────────────────────────────────────────
# Test 3: 2회 Trial 완주 시뮬레이션
# ─────────────────────────────────────────────
def test_two_trials_complete_without_error():
    """
    Optuna가 gate_rm 하이퍼파라미터 조합 2회 Trial을
    에러 없이 완주하는지 확인 (GateRMModel 포워드 포함)
    """
    n_num = 4
    B = 8
    C = 16  # candidate pool size

    def objective(trial: optuna.Trial) -> float:
        params = sample_gate_rm_params(trial)
        
        model = GateRMModel(
            n_num_features=n_num,
            n_bin_features=0,
            cat_cardinalities=[],
            n_classes=None,  # regression
            d_embedding=params["d_embedding"],
            n_layers=params["n_layers"],
            n_ensembles=params["n_ensembles"],
            context_dropout=params["context_dropout"],
        )
        model.eval()

        x_ = {'num': torch.randn(B, n_num)}
        candidate_x_ = {'num': torch.randn(C, n_num)}

        with torch.no_grad():
            out = model(
                x_=x_,
                y=None,
                candidate_x_=candidate_x_,
                candidate_y=torch.randn(C),
                context_size=8,
                is_train=False,
            )
        # 더미 손실 (RMSE 흉내)
        loss = out.squeeze().pow(2).mean().sqrt().item()
        return loss

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=2)

    assert len(study.trials) == 2, f"Trial 수 불일치: {len(study.trials)}"
    for trial in study.trials:
        assert trial.state == optuna.trial.TrialState.COMPLETE, (
            f"Trial이 완주되지 않음: {trial.state}"
        )


# ─────────────────────────────────────────────
# Test 4: GateRMModel — HPO 극단 파라미터 내성
# ─────────────────────────────────────────────
@pytest.mark.parametrize("d_embedding,n_layers,n_ensembles", [
    (8,  1, 2),   # 최솟값 조합
    (32, 4, 8),   # 최댓값 조합
    (16, 2, 4),   # 중간값 조합
])
def test_gate_rm_hpo_extreme_configs(d_embedding, n_layers, n_ensembles):
    """
    HPO 탐색 범위 내 극단 파라미터 조합에서도
    GateRMModel이 정상 출력을 반환하는지 검증
    """
    model = GateRMModel(
        n_num_features=5,
        n_bin_features=0,
        cat_cardinalities=[],
        n_classes=None,
        d_embedding=d_embedding,
        n_layers=n_layers,
        n_ensembles=n_ensembles,
        context_dropout=0.0,
    )
    model.eval()

    x_ = {'num': torch.randn(4, 5)}
    candidate_x_ = {'num': torch.randn(10, 5)}

    with torch.no_grad():
        out = model(
            x_=x_,
            y=None,
            candidate_x_=candidate_x_,
            candidate_y=torch.randn(10),
            context_size=5,
            is_train=False,
        )

    assert not torch.isnan(out).any(), (
        f"NaN 발생 (d_emb={d_embedding}, n_layers={n_layers}, n_ens={n_ensembles})"
    )
    assert out.shape[0] == 4, f"배치 차원 불일치: {out.shape}"


# ─────────────────────────────────────────────
# Test 5: Pruning(조기 종료) 핸들링
# ─────────────────────────────────────────────
def test_optuna_pruning_exception_handled():
    """
    Trial 내에서 TrialPruned 예외가 발생할 때
    Optuna가 해당 Trial을 PRUNED 상태로 처리하는지 확인
    """
    def objective_that_prunes(trial: optuna.Trial) -> float:
        # 의도적으로 첫 번째 trial을 프루닝
        raise optuna.exceptions.TrialPruned()

    study = optuna.create_study(direction="minimize")
    study.optimize(objective_that_prunes, n_trials=1)

    assert study.trials[0].state == optuna.trial.TrialState.PRUNED, (
        "Pruned Trial이 PRUNED 상태로 처리되지 않음"
    )


# ─────────────────────────────────────────────
# Test 6: Best Trial 파라미터 접근
# ─────────────────────────────────────────────
def test_optuna_best_trial_accessible():
    """
    2회 Trial 완주 후 best_trial 파라미터가
    정상적으로 접근 가능한지 확인
    """
    n_num = 3

    def objective(trial: optuna.Trial) -> float:
        params = sample_gate_rm_params(trial)
        return params["d_embedding"] * 0.01  # 더미 목적함수

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=2)

    best = study.best_trial
    assert best is not None
    assert "d_embedding" in best.params
    assert "n_layers" in best.params
    assert "n_ensembles" in best.params


# ─────────────────────────────────────────────
# Test 7: NaN 발산 Trial 자동 폐기
# ─────────────────────────────────────────────
def test_nan_loss_trial_handled():
    """
    목적함수가 NaN을 반환할 때 Optuna가 해당 Trial을
    FAIL 상태로 처리하는지 확인 (HPO 안정성)
    """
    import math

    def objective_nan(trial: optuna.Trial) -> float:
        return float('nan')

    study = optuna.create_study(direction="minimize")
    study.optimize(objective_nan, n_trials=1)

    assert study.trials[0].state == optuna.trial.TrialState.FAIL, (
        "NaN 반환 Trial이 FAIL 상태로 처리되지 않음"
    )


if __name__ == '__main__':
    test_optuna_importable_and_study_creation()
    test_search_space_sampling_valid()
    test_two_trials_complete_without_error()
    test_gate_rm_hpo_extreme_configs(8, 1, 2)
    test_gate_rm_hpo_extreme_configs(32, 4, 8)
    test_gate_rm_hpo_extreme_configs(16, 2, 4)
    test_optuna_pruning_exception_handled()
    test_optuna_best_trial_accessible()
    test_nan_loss_trial_handled()
    print("Phase 9 Optuna HPO 통합 테스트 전체 통과!")
