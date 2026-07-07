```python
import optuna
import torch
import torch.nn as nn
import torch.optim as optim

# 가상의 GateRMModel 클래스 임포트 (실제로는 lib.gate_rm 에서 임포트)
from lib.gate_rm import GateRMModel

def objective(trial):
    """
    Optuna HPO 최적화를 위한 Objective 함수
    TabR 오리지널 논문(Table 15) 가이드라인을 엄격히 준수한 Search Space 적용
    """
    
    # 1. 고정 통제 변인 (Search Space 제외)
    context_size = 96
    patience = 16
    batch_size = 256
    
    # 2. GateR-M 전용 하이퍼파라미터 탐색 범위 (Search Space)
    
    # d_embedding: 차원 수의 하한선을 96으로 명확히 규정하여 표현력(Capacity) 보장
    d_embedding = trial.suggest_int("d_embedding", 96, 384)
    
    # n_layers: Deep Retrieval-Augmented Layers의 적층 수
    n_layers = trial.suggest_int("n_layers", 1, 4)
    
    # n_ensembles: Parameter-Efficient Packed Ensemble 헤드 개수
    n_ensembles = trial.suggest_int("n_ensembles", 4, 8)
    
    # context_dropout (ffn_dropout 동일 취급): 드롭아웃 확률
    context_dropout = trial.suggest_float("context_dropout", 0.0, 0.6)
    
    # Optimizer 파라미터
    lr = trial.suggest_float("lr", 3e-5, 1e-3, log=True)
    
    # weight_decay: 0 또는 1e-6 ~ 1e-3 사이의 log-uniform 값
    # Categorical로 0.0을 포함시킬 수 있습니다.
    use_weight_decay = trial.suggest_categorical("use_weight_decay", [False, True])
    if use_weight_decay:
        weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
    else:
        weight_decay = 0.0

    # 3. 모델 아키텍처 인스턴스화
    # 위에서 정의한 조건들을 엄격하게 반영하여 GateR-M 모델 생성
    model = GateRMModel(
        n_num_features=8,           # 예: California Housing 수치형 피처 개수
        n_bin_features=0,
        cat_cardinalities=[],
        n_classes=None,             # Regression 문제이므로 None
        d_embedding=d_embedding,    # 제안된 임베딩 차원 (>= 96)
        n_layers=n_layers,          # 제안된 레이어 수
        n_ensembles=n_ensembles,    # 제안된 앙상블 헤드 수
        context_dropout=context_dropout,
        share_weights=False         # Feature-wise 분할 투영을 위해 False 유지
    )
    
    # Device 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # 기본 최적화 알고리즘: AdamW 고정
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    # 손실 함수: Regression 문제이므로 MSE 고정
    criterion = nn.MSELoss()

    # --- 여기서부터는 일반적인 PyTorch 학습 루프 (Training Loop) ---
    best_val_rmse = float("inf")
    epochs_without_improvement = 0
    
    for epoch in range(100): # max_epochs
        # (1) 학습 코드 (Train)
        model.train()
        # for x_anchor, x_neighbors, y in train_loader: ...
        
        # (2) 검증 코드 (Validation)
        model.eval()
        val_loss = 0.0
        # for x_anchor, x_neighbors, y in val_loader: ...
        
        val_rmse = val_loss ** 0.5
        
        # Optuna에 현재 Epoch의 점수 보고 (Pruning 지원용)
        trial.report(val_rmse, epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()

        # 조기 종료 조건 (Early Stopping)
        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                break
                
    # 최종 검증 점수 반환 (이 값을 최소화하는 것이 Optuna의 목표)
    return best_val_rmse

if __name__ == "__main__":
    # Optuna 스터디 생성 및 실행 (최소화 목표)
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=30)
    
    print("Best trial:")
    trial = study.best_trial
    print(f"  Value (RMSE): {trial.value}")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")
```
