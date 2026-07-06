import os
import sys
import torch
import pytest

project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ['PROJECT_DIR'] = project_dir
sys.path.append(project_dir)

from lib.gate_rm import FeatureWiseProjection

def test_feature_wise_projection():
    B, M, K, d = 4, 8, 15, 16
    
    # Test 1: Feature-wise (share_weights=False)
    proj_fw = FeatureWiseProjection(d_embedding=d, n_features=K, share_weights=False)
    x_anchor = torch.randn(B, K, d, requires_grad=True)
    x_neighbors = torch.randn(B, M, K, d, requires_grad=True)
    
    Q, K_out, V = proj_fw(x_anchor, x_neighbors)
    
    assert Q.shape == (B, K, d)
    assert K_out.shape == (B, M, K, d)
    assert V.shape == (B, M, K, d)
    
    # Backpropagation check
    loss = Q.sum() + K_out.sum() + V.sum()
    loss.backward()
    
    assert x_anchor.grad is not None
    assert x_neighbors.grad is not None
    print("Feature-wise projection test: PASS")

    # Test 2: Shared weights (share_weights=True)
    proj_shared = FeatureWiseProjection(d_embedding=d, n_features=K, share_weights=True)
    x_anchor = torch.randn(B, K, d, requires_grad=True)
    x_neighbors = torch.randn(B, M, K, d, requires_grad=True)
    
    Q, K_out, V = proj_shared(x_anchor, x_neighbors)
    
    assert Q.shape == (B, K, d)
    assert K_out.shape == (B, M, K, d)
    assert V.shape == (B, M, K, d)
    
    loss = Q.sum() + K_out.sum() + V.sum()
    loss.backward()
    
    assert x_anchor.grad is not None
    assert x_neighbors.grad is not None
    print("Shared weights projection test: PASS")

    # Test 3: Input shape validation check
    with pytest.raises(AssertionError):
        proj_fw(torch.randn(B, K), x_neighbors)  # 2D input should fail

    with pytest.raises(AssertionError):
        proj_fw(x_anchor, torch.randn(B, M, K))  # 3D input should fail
    print("Validation decorator test: PASS")

if __name__ == '__main__':
    test_feature_wise_projection()
