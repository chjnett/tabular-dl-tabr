import os
import sys
import torch
import pytest

project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ['PROJECT_DIR'] = project_dir
sys.path.append(project_dir)

from lib.gate_rm import StackedGateRRetrieval

def test_stacked_gate_r_retrieval():
    B, M, K, d = 2, 4, 3, 8
    n_layers = 3
    stacked_retrieval = StackedGateRRetrieval(
        d_embedding=d, n_features=K, context_dropout=0.0, n_layers=n_layers
    )
    
    x_anchor = torch.ones(B, K, d, requires_grad=True)
    x_neighbors = torch.ones(B, M, K, d, requires_grad=True)
    
    out, trajectories = stacked_retrieval(x_anchor, x_neighbors, return_trajectories=True)
    
    assert out.shape == (B, K, d)
    assert len(trajectories) == n_layers + 1
    assert trajectories[0] is x_anchor
    
    # Backpropagation check through deep stack
    loss = out.sum()
    loss.backward()
    
    assert x_anchor.grad is not None
    assert x_neighbors.grad is not None
    print("StackedGateRRetrieval test: PASS")

if __name__ == '__main__':
    test_stacked_gate_r_retrieval()
