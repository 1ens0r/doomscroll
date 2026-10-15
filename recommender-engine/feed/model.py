"""Intuition: A ~100k-param two-tower net where user habit and video quality are embedded
separately, fused with their interaction and context, then read out by five sigmoid heads."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent.parent
ART = ROOT / "artifacts"
HEADS = ["click", "like", "follow", "forward", "watch"]


def load_meta() -> dict:
    with open(ART / "feature_list.json") as f:
        return json.load(f)


def block_indices(dense: list[str]) -> dict[str, list[int]]:
    """Dense-input column slices per tower, derived from the feature list (never hardcoded)."""
    def pick(prefix: str) -> list[int]:
        return [i for i, f in enumerate(dense) if f.startswith(prefix)]

    return {
        "user": pick("u_"),
        "item": pick("i_"),
        "pair": pick("uc_") + pick("ui_") + pick("ctx_") + pick("cold_"),
    }


def _mlp(dims: list[int], dropout: float) -> nn.Sequential:
    layers: list[nn.Module] = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2:
            layers += [nn.ReLU(), nn.Dropout(dropout)]
    return nn.Sequential(*layers)


class TasteNet(nn.Module):
    """Two-tower ranker: returns raw logits for [click, like, follow, forward, watch]."""

    def __init__(self, blocks: dict[str, list[int]], cat_rows: int, model_cfg: dict) -> None:
        super().__init__()
        d = float(model_cfg["dropout"])
        emb = int(model_cfg["cat_emb_dim"])
        uh = list(model_cfg["user_hidden"])
        vh = list(model_cfg["video_hidden"])
        ph = list(model_cfg["pair_hidden"])
        fh = list(model_cfg["fusion_hidden"])
        self.register_buffer("idx_user", torch.tensor(blocks["user"], dtype=torch.long))
        self.register_buffer("idx_item", torch.tensor(blocks["item"], dtype=torch.long))
        self.register_buffer("idx_pair", torch.tensor(blocks["pair"], dtype=torch.long))
        self.cat_emb = nn.Embedding(cat_rows, emb)
        self.user_tower = _mlp([len(blocks["user"]), *uh], d)
        self.video_tower = _mlp([len(blocks["item"]) + emb, *vh], d)
        self.pair_block = _mlp([len(blocks["pair"]), *ph], d)
        fused = uh[-1] * 3 + ph[-1]
        self.fusion = _mlp([fused, *fh, len(HEADS)], d)

    def forward(self, x: torch.Tensor, cat: torch.Tensor) -> torch.Tensor:
        u = self.user_tower(x[:, self.idx_user])
        v = self.video_tower(torch.cat([x[:, self.idx_item], self.cat_emb(cat)], dim=1))
        p = self.pair_block(x[:, self.idx_pair])
        return self.fusion(torch.cat([u, v, u * v, p], dim=1))


@torch.no_grad()
def predict_probs(model: nn.Module, x: np.ndarray, cat: np.ndarray,
                  batch_size: int = 65536) -> np.ndarray:
    """Sigmoid probabilities (n, 5) in one batched pass; model is left in eval mode."""
    model.eval()
    out = np.empty((len(x), len(HEADS)), dtype=np.float32)
    for i in range(0, len(x), batch_size):
        xb = torch.from_numpy(np.ascontiguousarray(x[i:i + batch_size]))
        cb = torch.from_numpy(cat[i:i + batch_size].astype(np.int64))
        out[i:i + batch_size] = torch.sigmoid(model(xb, cb)).numpy()
    return out


def score_from_probs(probs: np.ndarray, weights: dict[str, float]) -> np.ndarray:
    w = np.array([weights[h] for h in HEADS], dtype=np.float32)
    return probs @ w


def build_model(meta: dict, policy: dict) -> TasteNet:
    return TasteNet(block_indices(meta["dense"]), int(meta["cat_embed_rows"]), policy["model"])


def n_params(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters()))
