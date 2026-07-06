import os
import sys
import torch
import numpy as np

# Add project root to sys.path and environment
project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ['PROJECT_DIR'] = project_dir
sys.path.append(project_dir)

import lib
from bin.tabr import Model, Config

# Monkeypatch Model.forward to use pure PyTorch cdist CPU fallback for search
original_forward = Model.forward

def patched_forward(self, *, x_, y, candidate_x_, candidate_y, context_size, is_train):
    with torch.set_grad_enabled(torch.is_grad_enabled() and not self.memory_efficient):
        candidate_k = (
            self._encode(candidate_x_)[1]
            if self.candidate_encoding_batch_size is None
            else torch.cat(
                [
                    self._encode(x)[1]
                    for x in delu.iter_batches(
                        candidate_x_, self.candidate_encoding_batch_size
                    )
                ]
            )
        )
    x, k = self._encode(x_)
    if is_train:
        assert y is not None
        candidate_k = torch.cat([k, candidate_k])
        candidate_y = torch.cat([y, candidate_y])
    else:
        assert y is None

    batch_size, d_main = k.shape
    device = k.device
    
    with torch.no_grad():
        # Pure PyTorch L2 search fallback to prevent segfault in faiss on macOS CPU
        if device.type == 'cpu':
            dists = torch.cdist(k, candidate_k, p=2.0)
            # squared L2 distance to match Faiss
            dists_sq = dists.square()
            k_neighbors = context_size + (1 if is_train else 0)
            distances, context_idx = torch.topk(dists_sq, k=k_neighbors, dim=-1, largest=False)
        else:
            import faiss
            import faiss.contrib.torch_utils
            if self.search_index is None:
                self.search_index = (
                    faiss.GpuIndexFlatL2(faiss.StandardGpuResources(), d_main)
                    if device.type == 'cuda'
                    else faiss.IndexFlatL2(d_main)
                )
            self.search_index.reset()
            self.search_index.add(candidate_k)
            distances, context_idx = self.search_index.search(
                k, context_size + (1 if is_train else 0)
            )
            
        if is_train:
            distances[
                context_idx == torch.arange(batch_size, device=device)[:, None]
            ] = torch.inf
            context_idx = context_idx.gather(-1, distances.argsort()[:, :-1])

    if self.memory_efficient and torch.is_grad_enabled():
        assert is_train
        context_k = self._encode(
            {
                ftype: torch.cat([x_[ftype], candidate_x_[ftype]])[
                    context_idx
                ].flatten(0, 1)
                for ftype in x_
            }
        )[1].reshape(batch_size, context_size, -1)
    else:
        context_k = candidate_k[context_idx]

    similarities = (
        -k.square().sum(-1, keepdim=True)
        + (2 * (k[..., None, :] @ context_k.transpose(-1, -2))).squeeze(-2)
        - context_k.square().sum(-1)
    )
    probs = torch.softmax(similarities, dim=-1)
    probs = self.dropout(probs)

    context_y_emb = self.label_encoder(candidate_y[context_idx][..., None])
    values = context_y_emb + self.T(k[:, None] - context_k)
    context_x = (probs[:, None] @ values).squeeze(1)
    x = x + context_x

    for block in self.blocks1:
        x = x + block(x)
    x = self.head(x)
    return x

Model.forward = patched_forward

def test_step():
    print("Loading config...")
    config_path = os.path.join(project_dir, "exp/debug/tabr_test.toml")
    config = lib.load_config(config_path)
    C = lib.make_config(Config, config)
    
    device = torch.device('cpu')
    print("Building dataset...")
    dataset = lib.build_dataset(**C.data).to_torch(device)
    
    print("Building model...")
    model = Model(
        n_num_features=dataset.n_num_features,
        n_bin_features=dataset.n_bin_features,
        cat_cardinalities=dataset.cat_cardinalities(),
        n_classes=dataset.n_classes(),
        **C.model,
    ).to(device)
    
    train_size = dataset.size('train')
    batch_idx = torch.arange(C.batch_size, device=device)
    
    def get_Xy(part: str, idx) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        batch = (
            {
                key[2:]: dataset.data[key][part]
                for key in dataset.data
                if key.startswith('X_')
            },
            dataset.Y[part],
        )
        return (
            batch
            if idx is None
            else ({k: v[idx] for k, v in batch[0].items()}, batch[1][idx])
        )
        
    x, y = get_Xy('train', batch_idx)
    candidate_indices = torch.arange(train_size, device=device)
    candidate_indices = candidate_indices[~torch.isin(candidate_indices, batch_idx)]
    candidate_x, candidate_y = get_Xy('train', candidate_indices)
    
    print("Running forward pass...")
    out = model(
        x_=x,
        y=y,
        candidate_x_=candidate_x,
        candidate_y=candidate_y,
        context_size=C.context_size,
        is_train=True,
    )
    print("Forward pass completed successfully! Output shape:", out.shape)

if __name__ == '__main__':
    test_step()
