import os
import sys
import torch
import pytest
import tempfile

project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ['PROJECT_DIR'] = project_dir
sys.path.append(project_dir)

from lib.gate_rm import GateRMModel

def test_extreme_input_handling():
    # Test 1: Zero categorical features
    model_no_cat = GateRMModel(
        n_num_features=5,
        n_bin_features=0,
        cat_cardinalities=[],
        n_classes=2,
        d_embedding=8,
        n_layers=1,
        n_ensembles=2,
    )
    x_num = {'num': torch.randn(2, 5)}
    out = model_no_cat(
        x_=x_num,
        y=torch.randint(0, 2, (2,)),
        candidate_x_={'num': torch.randn(5, 5)},
        candidate_y=torch.randint(0, 2, (5,)),
        context_size=3,
        is_train=True,
        ensemble_idx=torch.tensor([0, 1])
    )
    # Output dim is 1 for binary classification (n_classes = 2) in this library
    assert out.shape == (2, 1)
    print("Extreme input (zero categories) test: PASS")

def test_checkpoint_restoration():
    model = GateRMModel(
        n_num_features=2,
        n_bin_features=0,
        cat_cardinalities=[],
        n_classes=None,
        d_embedding=4,
        n_layers=1,
        n_ensembles=2,
        context_dropout=0.0, # Disable dropout for deterministic output comparison
    )
    model.eval() # Put in eval mode to disable any other stochastic behavior
    
    x_ = {'num': torch.randn(2, 2)}
    y = torch.randn(2)
    candidate_x_ = {'num': torch.randn(5, 2)}
    candidate_y = torch.randn(5)
    
    out1 = model(
        x_=x_,
        y=y,
        candidate_x_=candidate_x_,
        candidate_y=candidate_y,
        context_size=3,
        is_train=False,
    )
    
    # Save checkpoint
    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = os.path.join(tmpdir, "model.pt")
        torch.save(model.state_dict(), ckpt_path)
        
        # Load checkpoint into new model instance
        model2 = GateRMModel(
            n_num_features=2,
            n_bin_features=0,
            cat_cardinalities=[],
            n_classes=None,
            d_embedding=4,
            n_layers=1,
            n_ensembles=2,
            context_dropout=0.0,
        )
        model2.load_state_dict(torch.load(ckpt_path, weights_only=True))
        model2.eval()
        
        out2 = model2(
            x_=x_,
            y=y,
            candidate_x_=candidate_x_,
            candidate_y=candidate_y,
            context_size=3,
            is_train=False,
        )
        
        assert torch.allclose(out1, out2)
    print("Checkpoint restoration test: PASS")

if __name__ == '__main__':
    test_extreme_input_handling()
    test_checkpoint_restoration()
