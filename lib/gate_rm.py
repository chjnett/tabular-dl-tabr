import math
from typing import Literal, Optional, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import delu
import lib

def validate_input_shapes(func):
    def wrapper(self, x_anchor: Tensor, x_neighbors: Tensor, *args, **kwargs):
        assert x_anchor.ndim == 3, f"x_anchor must be 3D [B, K, d], got {x_anchor.ndim}D"
        assert x_neighbors.ndim == 4, f"x_neighbors must be 4D [B, M, K, d], got {x_neighbors.ndim}D"
        assert x_anchor.shape[1] == self.n_features, f"x_anchor features {x_anchor.shape[1]} != {self.n_features}"
        assert x_neighbors.shape[2] == self.n_features, f"x_neighbors features {x_neighbors.shape[2]} != {self.n_features}"
        assert x_anchor.shape[2] == self.d_embedding, f"x_anchor embedding dim {x_anchor.shape[2]} != {self.d_embedding}"
        assert x_neighbors.shape[3] == self.d_embedding, f"x_neighbors embedding dim {x_neighbors.shape[3]} != {self.d_embedding}"
        return func(self, x_anchor, x_neighbors, *args, **kwargs)
    return wrapper

class FeatureWiseProjection(nn.Module):
    def __init__(self, d_embedding: int, n_features: int, share_weights: bool = False):
        super().__init__()
        self.d_embedding = d_embedding
        self.n_features = n_features
        self.share_weights = share_weights
        
        if share_weights:
            self.W_Q = nn.Linear(d_embedding, d_embedding, bias=False)
            self.W_K = nn.Linear(d_embedding, d_embedding, bias=False)
            self.W_V = nn.Linear(d_embedding, d_embedding, bias=False)
        else:
            self.W_Q = nn.Parameter(torch.empty(n_features, d_embedding, d_embedding))
            self.W_K = nn.Parameter(torch.empty(n_features, d_embedding, d_embedding))
            self.W_V = nn.Parameter(torch.empty(n_features, d_embedding, d_embedding))
            self.reset_parameters()

        self.norm_q = nn.LayerNorm(d_embedding)
        self.norm_k = nn.LayerNorm(d_embedding)
        self.norm_v = nn.LayerNorm(d_embedding)

    def reset_parameters(self):
        if not self.share_weights:
            for W in [self.W_Q, self.W_K, self.W_V]:
                bound = 1.0 / math.sqrt(self.d_embedding)
                nn.init.uniform_(W, -bound, bound)

    @validate_input_shapes
    def forward(self, x_anchor: Tensor, x_neighbors: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        if self.share_weights:
            Q = self.W_Q(x_anchor)
            K_out = self.W_K(x_neighbors)
            V = self.W_V(x_neighbors)
        else:
            Q = torch.einsum('bkd,kde->bke', x_anchor, self.W_Q)
            K_out = torch.einsum('bmkd,kde->bmke', x_neighbors, self.W_K)
            V = torch.einsum('bmkd,kde->bmke', x_neighbors, self.W_V)
            
        return self.norm_q(Q), self.norm_k(K_out), self.norm_v(V)

class GateRRetrieval(nn.Module):
    def __init__(self, d_embedding: int, n_features: int, context_dropout: float, share_weights: bool = False):
        super().__init__()
        self.projection = FeatureWiseProjection(d_embedding, n_features, share_weights)
        self.dropout = nn.Dropout(context_dropout)
        self.d_embedding = d_embedding

    def forward(self, x_anchor: Tensor, x_neighbors: Tensor, label_emb: Optional[Tensor] = None, return_attn: bool = False) -> Union[Tensor, tuple[Tensor, Tensor]]:
        # x_anchor: [B, K, d]
        # x_neighbors: [B, M, K, d] (M = context_size)
        # label_emb: [B, M, K, d] (optional, neighbor label embeddings)
        Q, K, V = self.projection(x_anchor, x_neighbors)
        
        # If label embeddings are provided, add them to values (TabR-style)
        if label_emb is not None:
            V = V + label_emb
        
        # Feature-wise matching score using inner product:
        # Q: [B, K, d], K: [B, M, K, d] -> scores: [B, M, K]
        scores = torch.einsum('bkd,bmkd->bmk', Q, K) / math.sqrt(self.d_embedding)
        
        # Softmax over the neighbor dimension (M)
        attn = F.softmax(scores, dim=1)
        attn = self.dropout(attn)
        
        # Values aggregation:
        # attn: [B, M, K], V: [B, M, K, d] -> Z: [B, K, d]
        Z = torch.einsum('bmk,bmkd->bkd', attn, V)
        
        # Skip connection
        out = x_anchor + Z
        
        if return_attn:
            return out, attn
        return out

class StackedGateRRetrieval(nn.Module):
    def __init__(self, d_embedding: int, n_features: int, context_dropout: float, n_layers: int = 3, share_weights: bool = False):
        super().__init__()
        self.n_layers = n_layers
        self.layers = nn.ModuleList([
            GateRRetrieval(d_embedding, n_features, context_dropout, share_weights)
            for _ in range(n_layers)
        ])
        self.norms = nn.ModuleList([
            nn.LayerNorm(d_embedding)
            for _ in range(n_layers)
        ])

    def forward(self, x_anchor: Tensor, x_neighbors: Tensor, label_emb: Optional[Tensor] = None, return_trajectories: bool = False) -> Union[Tensor, tuple[Tensor, list[Tensor]]]:
        h = x_anchor
        trajectories = [h]
        
        for norm, layer in zip(self.norms, self.layers):
            # Pre-LayerNorm residual connection
            h_norm = norm(h)
            h = layer(h_norm, x_neighbors, label_emb=label_emb)
            trajectories.append(h)
            
        if return_trajectories:
            return h, trajectories
        return h

class BatchEnsembleLinear(nn.Module):
    def __init__(self, d_in: int, d_out: int, n_ensembles: int, bias: bool = True):
        super().__init__()
        self.d_in = d_in
        self.d_out = d_out
        self.n_ensembles = n_ensembles
        
        self.weight = nn.Parameter(torch.empty(d_out, d_in))
        self.bias = nn.Parameter(torch.empty(d_out)) if bias else None
        
        self.r = nn.Parameter(torch.empty(n_ensembles, d_in))
        self.s = nn.Parameter(torch.empty(n_ensembles, d_out))
        
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)
            
        nn.init.ones_(self.r)
        nn.init.ones_(self.s)

    def forward(self, x: Tensor, ensemble_idx: Optional[Tensor] = None) -> Tensor:
        if ensemble_idx is not None:
            r_w = self.r[ensemble_idx]
            s_w = self.s[ensemble_idx]
            out = (x * r_w) @ self.weight.t()
            if self.bias is not None:
                out = out + self.bias
            return out * s_w
        else:
            x_ens = x.unsqueeze(1) * self.r.unsqueeze(0)
            out = torch.matmul(x_ens, self.weight.t())
            if self.bias is not None:
                out = out + self.bias.unsqueeze(0).unsqueeze(0)
            out = out * self.s.unsqueeze(0)
            return out

class FeatureEmbedder(nn.Module):
    def __init__(
        self,
        *,
        n_num_features: int,
        n_bin_features: int,
        cat_cardinalities: list[int],
        d_embedding: int,
        num_embeddings: Optional[dict] = None,
    ):
        super().__init__()
        self.n_num_features = n_num_features
        self.n_bin_features = n_bin_features
        self.cat_cardinalities = cat_cardinalities
        self.d_embedding = d_embedding
        
        if n_num_features > 0:
            if num_embeddings is not None:
                self.num_emb = lib.make_module(num_embeddings, n_features=n_num_features)
                self.num_proj = nn.Linear(num_embeddings['d_embedding'], d_embedding)
            else:
                self.num_emb = None
                self.num_proj = nn.Linear(1, d_embedding)
                
        all_cat_cardinalities = [2] * n_bin_features + cat_cardinalities
        if all_cat_cardinalities:
            self.cat_emb = nn.ModuleList([
                nn.Embedding(cardinality, d_embedding)
                for cardinality in all_cat_cardinalities
            ])
        else:
            self.cat_emb = None

    def forward(self, x_: dict[str, Tensor]) -> Tensor:
        embeddings = []
        
        x_num = x_.get('num')
        if x_num is not None:
            if self.num_emb is not None:
                num_e = self.num_emb(x_num)
                num_e = self.num_proj(num_e)
                embeddings.append(num_e)
            else:
                num_e = x_num.unsqueeze(-1)
                num_e = self.num_proj(num_e)
                embeddings.append(num_e)
                
        x_bin = x_.get('bin')
        x_cat = x_.get('cat')
        cat_tensors = []
        if x_bin is not None:
            cat_tensors.append(x_bin)
        if x_cat is not None:
            cat_tensors.append(x_cat)
            
        if cat_tensors:
            x_cat_all = torch.cat(cat_tensors, dim=-1)
            for i, emb_layer in enumerate(self.cat_emb):
                col = x_cat_all[:, i]
                col_e = emb_layer(col).unsqueeze(1)
                embeddings.append(col_e)
                
        return torch.cat(embeddings, dim=1)

class GateRMModel(nn.Module):
    def __init__(
        self,
        *,
        n_num_features: int,
        n_bin_features: int,
        cat_cardinalities: list[int],
        n_classes: Optional[int],
        d_embedding: int,
        n_layers: int = 3,
        n_ensembles: int = 4,
        context_dropout: float = 0.2,
        share_weights: bool = False,
        num_embeddings: Optional[dict] = None,
    ):
        super().__init__()
        self.n_features = n_num_features + n_bin_features + len(cat_cardinalities)
        self.d_embedding = d_embedding
        self.n_classes = n_classes
        self.n_ensembles = n_ensembles
        
        self.embedder = FeatureEmbedder(
            n_num_features=n_num_features,
            n_bin_features=n_bin_features,
            cat_cardinalities=cat_cardinalities,
            d_embedding=d_embedding,
            num_embeddings=num_embeddings,
        )
        
        self.retrieval = StackedGateRRetrieval(
            d_embedding=d_embedding,
            n_features=self.n_features,
            context_dropout=context_dropout,
            n_layers=n_layers,
            share_weights=share_weights,
        )
        
        # >>> Label Retrieval (TabR core mechanism)
        # Label encoder: maps neighbor labels to d_embedding space
        self.label_encoder = (
            nn.Linear(1, d_embedding)
            if n_classes is None  # regression
            else nn.Sequential(
                nn.Embedding(n_classes, d_embedding),
                delu.nn.Lambda(lambda x: x.squeeze(-2)),
            )
        )
        # Label transform: processes (anchor - neighbor) difference
        d_block = d_embedding * 2
        self.label_transform = nn.Sequential(
            nn.Linear(d_embedding, d_block),
            nn.ReLU(),
            nn.Dropout(context_dropout),
            nn.Linear(d_block, d_embedding, bias=False),
        )
        
        d_in_predictor = self.n_features * d_embedding
        d_out = lib.get_d_out(n_classes)
        self.predictor = BatchEnsembleLinear(d_in_predictor, d_out, n_ensembles)
        
        self._reset_label_parameters()

    def _reset_label_parameters(self):
        if isinstance(self.label_encoder, nn.Linear):
            bound = 1 / math.sqrt(2.0)
            nn.init.uniform_(self.label_encoder.weight, -bound, bound)
            nn.init.uniform_(self.label_encoder.bias, -bound, bound)
        else:
            assert isinstance(self.label_encoder[0], nn.Embedding)
            nn.init.uniform_(self.label_encoder[0].weight, -1.0, 1.0)

    def forward(
        self,
        *,
        x_: dict[str, Tensor],
        y: Optional[Tensor],
        candidate_x_: dict[str, Tensor],
        candidate_y: Tensor,
        context_size: int,
        is_train: bool,
        ensemble_idx: Optional[Tensor] = None,
    ) -> Tensor:
        x_anchor = self.embedder(x_)
        candidate_x = self.embedder(candidate_x_)
        
        batch_size = x_anchor.shape[0]
        device = x_anchor.device
        
        k_anchor = x_anchor.flatten(1)
        k_candidate = candidate_x.flatten(1)
        
        if is_train:
            assert y is not None
            k_candidate = torch.cat([k_anchor, k_candidate])
            candidate_y = torch.cat([y, candidate_y])
        
        dists = torch.cdist(k_anchor, k_candidate, p=2.0)
        dists_sq = dists.square()
        
        k_neighbors = context_size + (1 if is_train else 0)
        distances, context_idx = torch.topk(dists_sq, k=k_neighbors, dim=-1, largest=False)
        
        if is_train:
            distances[
                context_idx == torch.arange(batch_size, device=device)[:, None]
            ] = torch.inf
            context_idx = context_idx.gather(-1, distances.argsort()[:, :-1])
            
        all_features = torch.cat([x_anchor, candidate_x], dim=0) if is_train else candidate_x
        x_neighbors = all_features[context_idx]  # [B, M, K, d]
        
        # >>> Label Retrieval: encode neighbor labels and compute difference transform
        # context_y: [B, M]
        context_y = candidate_y[context_idx]  # [B, M]
        # context_y_emb: [B, M, d_embedding]
        context_y_emb = self.label_encoder(context_y[..., None])  # regression: [B, M, 1] -> [B, M, d]
        
        # Anchor-neighbor difference in flattened key space, projected per-feature
        # k_anchor: [B, K*d], k_candidate[context_idx]: [B, M, K*d]
        context_k = k_candidate[context_idx]  # [B, M, K*d]
        diff = k_anchor[:, None, :] - context_k  # [B, M, K*d]
        # Reshape to per-feature: [B, M, K, d]
        diff = diff.reshape(batch_size, context_size, self.n_features, self.d_embedding)
        # Transform the difference: [B, M, K, d]
        diff_transformed = self.label_transform(diff)
        # Combine label embedding (broadcast over features) + difference
        # context_y_emb: [B, M, d] -> [B, M, 1, d] (broadcast over K features)
        label_emb = context_y_emb.unsqueeze(2) + diff_transformed  # [B, M, K, d]
        
        h = self.retrieval(x_anchor, x_neighbors, label_emb=label_emb)
        h_flat = h.flatten(1)
        
        out = self.predictor(h_flat, ensemble_idx)
        return out
