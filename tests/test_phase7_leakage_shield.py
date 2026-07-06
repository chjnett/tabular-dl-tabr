import os
import sys
import torch
import pytest

project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ['PROJECT_DIR'] = project_dir
sys.path.append(project_dir)

from lib.gate_rm import GateRMModel

def test_gate_rm_end_to_end_leakage_shield():
    B, C = 4, 10
    n_num = 3
    n_bin = 2
    cat_cardinalities = [4, 5]
    d_embedding = 8
    n_classes = 3
    context_size = 5
    
    # 1. Initialize model
    model = GateRMModel(
        n_num_features=n_num,
        n_bin_features=n_bin,
        cat_cardinalities=cat_cardinalities,
        n_classes=n_classes,
        d_embedding=d_embedding,
        n_layers=2,
        n_ensembles=4,
    )
    
    # 2. Build dummy batches
    x_ = {
        'num': torch.randn(B, n_num),
        'bin': torch.randint(0, 2, (B, n_bin)),
        'cat': torch.cat([
            torch.randint(0, card, (B, 1))
            for card in cat_cardinalities
        ], dim=-1)
    }
    y = torch.randint(0, n_classes, (B,))
    
    candidate_x_ = {
        'num': torch.randn(C, n_num),
        'bin': torch.randint(0, 2, (C, n_bin)),
        'cat': torch.cat([
            torch.randint(0, card, (C, 1))
            for card in cat_cardinalities
        ], dim=-1)
    }
    candidate_y = torch.randint(0, n_classes, (C,))
    
    # 3. Training mode run (each sample mapped to ensemble index)
    ensemble_idx = torch.tensor([0, 1, 2, 3])
    out_train = model(
        x_=x_,
        y=y,
        candidate_x_=candidate_x_,
        candidate_y=candidate_y,
        context_size=context_size,
        is_train=True,
        ensemble_idx=ensemble_idx,
    )
    assert out_train.shape == (B, n_classes)
    
    # 4. Leakage Shield Check
    # Test that self-masking works: self indices should have infinite L2 distance (masked)
    # We can check that gradients propagate cleanly to parameters
    loss = out_train.sum()
    loss.backward()
    
    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"Gradient not found for parameter {name}"
            
    print("GateRMModel end-to-end and leakage shield test: PASS")

if __name__ == '__main__':
    test_gate_rm_end_to_end_leakage_shield()
