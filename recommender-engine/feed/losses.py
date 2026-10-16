"""Intuition: One objective per ablation arm behind a single callable, so the trainer can swap
L0-L7 losses without touching the model, the data, or the evaluation protocol. `masked_bce` is the
canonical L0 and is proven equal to `feed.train.masked_head_loss` by a parity test."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from feed.model import HEADS
from feed.train import masked_head_loss


def _head_losses(bce: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
    denom = m.sum(dim=0).clamp(min=1.0)
    return (bce * m).sum(dim=0) / denom


def masked_bce(logits: torch.Tensor, y: torch.Tensor, m: torch.Tensor,
               pos_w: torch.Tensor, ctx: dict) -> dict:
    """L0 / L2: masked per-head BCE with class-balanced positive weights `pos_w`."""
    bce = F.binary_cross_entropy_with_logits(logits, y, pos_weight=pos_w, reduction="none")
    per_head = _head_losses(bce, m)
    total = (per_head * ctx["head_w"]).sum()
    return {"total": total, "per_head": per_head.detach()}


def equal_head_bce(logits: torch.Tensor, y: torch.Tensor, m: torch.Tensor,
                   pos_w: torch.Tensor, ctx: dict) -> dict:
    """L4: identical to L0 but with all head weights forced to 1 (drop the hand-tuned balance)."""
    bce = F.binary_cross_entropy_with_logits(logits, y, pos_weight=pos_w, reduction="none")
    per_head = _head_losses(bce, m)
    return {"total": per_head.sum(), "per_head": per_head.detach()}


class UncertaintyWeighting(nn.Module):
    """L4 (Kendall et al.): learn log-variance per head so hard heads are down-weighted automatically.
    total = sum_j per_head_j / (2 exp(2 s_j)) + s_j, with s_j the learnable parameter."""

    def __init__(self, n_heads: int) -> None:
        super().__init__()
        self.log_sigma = nn.Parameter(torch.zeros(n_heads))

    def forward(self, logits: torch.Tensor, y: torch.Tensor, m: torch.Tensor,
                pos_w: torch.Tensor, ctx: dict) -> dict:
        bce = F.binary_cross_entropy_with_logits(logits, y, pos_weight=pos_w, reduction="none")
        per_head = _head_losses(bce, m)
        weighted = per_head / (2.0 * torch.exp(2.0 * self.log_sigma)) + self.log_sigma
        return {"total": weighted.sum(), "per_head": per_head.detach()}


class LambdaRankLoss:
    """L3: graded pairwise logistic loss on tier, within one batch's per-user impression groups
    only. Pair weight = |gain_i - gain_j| (graded BPR on tiers). Requires grouped batching."""

    needs_groups = True

    def __init__(self, gains: list[int], scale: float = 1.0, max_group: int = 64) -> None:
        self.gains = torch.tensor(gains, dtype=torch.float32)
        self.scale = float(scale)
        self.max_group = int(max_group)

    def __call__(self, logits: torch.Tensor, y: torch.Tensor, m: torch.Tensor,
                 pos_w: torch.Tensor, ctx: dict) -> dict:
        probs = torch.sigmoid(logits)
        score = (probs * ctx["score_w"]).sum(dim=1)
        uid = ctx["uid"]
        tier = ctx["tier"].long()
        gains = self.gains.to(logits.device)
        order = torch.argsort(uid, stable=True)
        us, ss, ts = uid[order], score[order], tier[order]
        change = torch.ones_like(us, dtype=torch.bool)
        change[1:] = us[1:] != us[:-1]
        starts = torch.nonzero(change).flatten().tolist()
        ends = starts[1:] + [len(us)]
        pair_sum = score.new_zeros(())
        npair = 0
        for a, b in zip(starts, ends):
            if b - a < 2:
                continue
            s = ss[a:min(b, a + self.max_group)]
            t = ts[a:min(b, a + self.max_group)]
            gt = (t[:, None] > t[None, :])
            if not bool(gt.any()):
                continue
            w = (gains[t][:, None] - gains[t][None, :]).abs()[gt]
            d = (s[:, None] - s[None, :])[gt]
            pair_sum = pair_sum + (w * F.softplus(-d)).sum()
            npair += int(gt.sum())
        total = pair_sum / max(npair, 1)
        with torch.no_grad():
            bce = F.binary_cross_entropy_with_logits(logits, y, pos_weight=pos_w, reduction="none")
            per_head = _head_losses(bce, m)
        return {"total": total * self.scale, "per_head": per_head.detach()}


class HybridLoss:
    """Promoted objective: masked per-head BCE (keeps all 5 heads calibrated) plus a weighted
    graded pairwise LambdaRank term on tier, so the shared trunk also learns the within-user
    ordering it is graded on. `rank_weight` trades the two off."""

    needs_groups = True

    def __init__(self, gains: list[int], rank_weight: float, max_group: int) -> None:
        self.rank = LambdaRankLoss(gains, 1.0, max_group)
        self.rank_weight = float(rank_weight)

    def __call__(self, logits: torch.Tensor, y: torch.Tensor, m: torch.Tensor,
                 pos_w: torch.Tensor, ctx: dict) -> dict:
        bce = masked_bce(logits, y, m, pos_w, ctx)
        rank = self.rank(logits, y, m, pos_w, ctx)
        return {"total": bce["total"] + self.rank_weight * rank["total"],
                "per_head": bce["per_head"]}


def build_loss(name: str, policy: dict) -> tuple:
    """Return (loss_fn, extra_trainable_params) for an ablation arm name."""
    name = (name or "masked_bce").lower()
    if name in ("masked_bce", "l0"):
        return masked_bce, []
    if name == "equal_heads":
        return equal_head_bce, []
    if name in ("uncertainty", "l4_uncertainty"):
        mod = UncertaintyWeighting(len(HEADS))
        return mod.forward, list(mod.parameters())
    if name in ("lambdarank", "l3"):
        lr = LambdaRankLoss(policy["_gains"],
                            scale=float(policy["train"].get("rank_scale", 1.0)),
                            max_group=int(policy["train"].get("rank_max_group", 64)))
        return lr, []
    if name == "hybrid":
        hy = HybridLoss(policy["_gains"], float(policy["train"].get("rank_weight", 0.5)),
                        int(policy["train"].get("rank_max_group", 64)))
        return hy, []
    raise ValueError(f"unknown loss {name!r}")


# --- diagnostics (L1, L7) ---------------------------------------------------------------------

def reliability(probs: np.ndarray, y: np.ndarray, m: np.ndarray | None = None,
                n_bins: int = 10) -> dict:
    """Binned confidence vs accuracy (reliability diagram) for one head."""
    p = np.asarray(probs, dtype=np.float64)
    t = np.asarray(y, dtype=np.float64)
    if m is not None:
        sel = np.asarray(m) > 0
        p, t = p[sel], t[sel]
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, n_bins - 1)
    counts = np.bincount(idx, minlength=n_bins)
    conf = np.bincount(idx, weights=p, minlength=n_bins) / np.maximum(counts, 1)
    acc = np.bincount(idx, weights=t, minlength=n_bins) / np.maximum(counts, 1)
    return {"counts": counts.tolist(), "confidence": conf.tolist(), "accuracy": acc.tolist(),
            "n": int(len(p))}


def ece(probs: np.ndarray, y: np.ndarray, m: np.ndarray | None = None, n_bins: int = 10) -> float:
    rel = reliability(probs, y, m, n_bins)
    counts = np.asarray(rel["counts"], dtype=np.float64)
    if counts.sum() == 0:
        return float("nan")
    gap = np.abs(np.asarray(rel["confidence"]) - np.asarray(rel["accuracy"]))
    return float((counts / counts.sum() * gap).sum())


def brier(probs: np.ndarray, y: np.ndarray, m: np.ndarray | None = None) -> float:
    p = np.asarray(probs, dtype=np.float64)
    t = np.asarray(y, dtype=np.float64)
    if m is not None:
        sel = np.asarray(m) > 0
        p, t = p[sel], t[sel]
    return float(np.mean((p - t) ** 2)) if len(p) else float("nan")


def log_loss(probs: np.ndarray, y: np.ndarray, m: np.ndarray | None = None) -> float:
    p = np.clip(np.asarray(probs, dtype=np.float64), 1e-9, 1 - 1e-9)
    t = np.asarray(y, dtype=np.float64)
    if m is not None:
        sel = np.asarray(m) > 0
        p, t = p[sel], t[sel]
    return float(-np.mean(t * np.log(p) + (1 - t) * np.log(1 - p))) if len(p) else float("nan")
