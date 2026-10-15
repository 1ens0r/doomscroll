"""Intuition: One training loop for the Taste model: masked per-head BCE with class-balanced
positive weights, OneCycle AdamW, and early stopping on validation NDCG@10."""
from __future__ import annotations

import time

import numpy as np
import torch
import torch.nn.functional as F

from feed.eval import per_user_ndcg
from feed.model import HEADS, n_params, predict_probs, score_from_probs


def compute_pos_weights(y: np.ndarray, m: np.ndarray, cap: float) -> np.ndarray:
    """pos_weight = min(sqrt(neg/pos), cap) per head, from observed rows only."""
    out = np.ones(y.shape[1], dtype=np.float32)
    for j in range(y.shape[1]):
        sel = m[:, j] > 0
        p = float(np.clip(y[sel, j].sum() / max(1.0, sel.sum()), 1e-9, 1.0 - 1e-9))
        out[j] = min(float(np.sqrt((1.0 - p) / p)), cap)
    return out


def masked_head_loss(logits: torch.Tensor, y: torch.Tensor, m: torch.Tensor,
                     pos_w: torch.Tensor, head_w: torch.Tensor) -> torch.Tensor:
    """BCE per head, zeroed where the action is unobserved, weight-summed over heads."""
    bce = F.binary_cross_entropy_with_logits(logits, y, pos_weight=pos_w, reduction="none")
    denom = m.sum(dim=0).clamp(min=1.0)
    head_loss = (bce * m).sum(dim=0) / denom
    return (head_loss * head_w).sum()


def _val_ndcg(model: torch.nn.Module, val: dict, policy: dict, gains: list[int],
              k: int, keep: np.ndarray) -> float:
    probs = predict_probs(model, val["x"], val["cat"])
    scores = score_from_probs(probs, policy["score_weights"])
    ecfg = policy["eval"]
    _, nd = per_user_ndcg(scores, val["tier"], val["user_id"], gains, k, keep=keep,
                          min_impressions=ecfg["min_impressions"], min_tiers=ecfg["min_tiers"])
    return float(nd.mean()) if len(nd) else 0.0


def train_taste(model: torch.nn.Module, train: dict, val: dict, policy: dict,
                gains: list[int], seed: int = 42, log=print) -> dict:
    """Train in-memory on the whole train split; returns history + best state info."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    tcfg, mcfg, ecfg = policy["train"], policy["model"], policy["eval"]
    head_w = torch.tensor([mcfg["loss_weights"][h] for h in HEADS], dtype=torch.float32)
    pos_w = torch.tensor(compute_pos_weights(train["y"], train["m"], mcfg["pos_weight_cap"]),
                         dtype=torch.float32)
    xt = torch.from_numpy(train["x"])
    ct = torch.from_numpy(train["cat"].astype(np.int64))
    yt = torch.from_numpy(train["y"])
    mt = torch.from_numpy(train["m"])
    val_feed = (val["is_search"] == 0)
    n, bs = len(xt), int(tcfg["batch_size"])
    steps = int(np.ceil(n / bs))
    opt = torch.optim.AdamW(model.parameters(), lr=tcfg["lr"],
                            weight_decay=tcfg["weight_decay"])
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=tcfg["lr"], total_steps=int(tcfg["max_epochs"]) * steps)
    history: list[dict] = []
    best = {"ndcg10": -1.0, "epoch": -1, "state": None}
    no_improve = 0
    for epoch in range(1, int(tcfg["max_epochs"]) + 1):
        model.train()
        t0 = time.time()
        perm = rng.permutation(n)
        running = 0.0
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            opt.zero_grad(set_to_none=True)
            loss = masked_head_loss(model(xt[idx], ct[idx]), yt[idx], mt[idx], pos_w, head_w)
            loss.backward()
            opt.step()
            sched.step()
            running += float(loss.detach()) * len(idx)
        m10 = _val_ndcg(model, val, policy, gains, ecfg["ndcg_ks"][0], val_feed)
        m20 = _val_ndcg(model, val, policy, gains, ecfg["ndcg_ks"][1], val_feed)
        history.append({"epoch": epoch, "train_loss": running / n,
                        "val_ndcg10": m10, "val_ndcg20": m20,
                        "seconds": time.time() - t0})
        log(f"[train] epoch {epoch}/{tcfg['max_epochs']} loss={running / n:.4f} "
            f"val ndcg@10={m10:.4f} ndcg@20={m20:.4f} ({time.time() - t0:.1f}s)")
        if m10 > best["ndcg10"] + 1e-5:
            best = {"ndcg10": m10, "epoch": epoch,
                    "state": {k: v.detach().clone() for k, v in model.state_dict().items()}}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= int(tcfg["early_stop_patience"]):
                log(f"[train] early stop: no val ndcg@10 gain for {no_improve} epochs")
                break
    if best["state"] is not None:
        model.load_state_dict(best["state"])
    return {"history": history, "best_epoch": best["epoch"], "best_val_ndcg10": best["ndcg10"],
            "pos_weights": [float(x) for x in pos_w], "n_params": n_params(model),
            "n_train_rows": int(n)}
