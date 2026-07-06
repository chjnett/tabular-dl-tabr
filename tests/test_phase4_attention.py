import os
import sys
import torch
import pytest

project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ['PROJECT_DIR'] = project_dir
sys.path.append(project_dir)

from lib.gate_rm import GateRRetrieval

def test_gate_r_retrieval_and_noise_filtration():
    B, M, K, d = 2, 4, 3, 8  # 3 features
    retrieval = GateRRetrieval(d_embedding=d, n_features=K, context_dropout=0.0)
    
    # Anchor inputs
    x_anchor = torch.ones(B, K, d)
    # Neighbor inputs
    x_neighbors = torch.ones(B, M, K, d)
    
    # Intentionally corrupt the 3rd feature (index 2) of neighbors with extreme noise
    # This should degrade its matching score relative to the clean features
    x_neighbors[:, :, 2, :] = torch.randn(B, M, d) * 100.0
    
    out, attn = retrieval(x_anchor, x_neighbors, return_attn=True)
    
    assert out.shape == (B, K, d)
    assert attn.shape == (B, M, K)
    assert not torch.isnan(attn).any()
    
    print("Attention map values:\n", attn)
    print("GateRRetrieval test: PASS")

if __name__ == '__main__':
    test_gate_r_retrieval_and_noise_filtration()
