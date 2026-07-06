import os
import sys
import torch
import pytest

project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ['PROJECT_DIR'] = project_dir
sys.path.append(project_dir)

from lib.gate_rm import BatchEnsembleLinear

def test_batch_ensemble_linear():
    B, d_in, d_out = 4, 8, 4
    n_ensembles = 4
    
    model = BatchEnsembleLinear(d_in, d_out, n_ensembles=n_ensembles)
    x = torch.randn(B, d_in, requires_grad=True)
    
    # 1. Training mode: map each sample to a specific member
    ens_idx = torch.tensor([0, 1, 2, 3])
    out_train = model(x, ens_idx)
    assert out_train.shape == (B, d_out)
    
    loss_train = out_train.sum()
    loss_train.backward()
    assert x.grad is not None
    assert model.r.grad is not None
    assert model.s.grad is not None
    
    # Reset grad
    x.grad.zero_()
    model.zero_grad()
    
    # 2. Inference mode: evaluate all members
    out_inf = model(x)
    assert out_inf.shape == (B, n_ensembles, d_out)
    
    loss_inf = out_inf.sum()
    loss_inf.backward()
    assert x.grad is not None
    assert model.r.grad is not None
    print("BatchEnsembleLinear tests: PASS")

if __name__ == '__main__':
    test_batch_ensemble_linear()
