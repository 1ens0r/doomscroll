"""Intuition: One training loop for the Taste model: masked per-head BCE with class-balanced
positive weights, a config-selected objective (L0-L7), OneCycle/cosine AdamW, per-epoch val loss,
and early stopping on validation NDCG@10. `masked_head_loss` here is the canonical L0 objective."""
from __future__ import annotations

import time

import numpy as np
import torch
import torch.nn.functional as F

from feed.eval import per_user_ndcg
from feed.model import HEADS, n_params, predict_probs, score_from_probs


def compute_pos_weights(y: np.ndarray, m: np.ndarray, cap: float,
                        mode: str = "sqrt") -> np.ndarray:
    """Observed-label weighting per head, from observed rows only.
    sqrt: min(sqrt(neg/pos), cap). none: 1.0. enn: effective-number ratio min(..., cap)."""
    out = np.ones(y.shape[1], dtype=np.float32)
    beta = float(np.clip(0.999, 0.0, 1.0 - 1e-6))
    for j in range(y.shape[1]):
        sel = m[:, j] > 0
        n_obs = float(sel.sum())
        n_pos = float(np.clip(y[sel, j].sum(), 0.0, n_obs))
        n_neg = max(n_obs - n_pos, 0.0)
        p = float(np.clip(n_pos / max(1.0, n_obs), 1e-9, 1.0 - 1e-9))
        if mode == "none":
            out[j] = 1.0
        elif mode == "enn":
            w_pos = (1.0 - beta) / (1.0 - beta ** max(n_pos, 1.0))
            w_neg = (1.0 - beta) / (1.0 - beta ** max(n_neg, 1.0))
            out[j] = min(float(w_pos / max(w_neg, 1e-12)), cap)
        else:
            out[j] = min(float(np.sqrt((1.0 - p) / p)), cap)
    return out


def masked_head_loss(logits: torch.Tensor, y: torch.Tensor, m: torch.Tensor,
                     pos_w: torch.Tensor, head_w: torch.Tensor) -> torch.Tensor:
    """BCE per head, zeroed where the action is unobserved, weight-summed over heads."""
    bce = F.binary_cross_entropy_with_logits(logits, y, pos_weight=pos_w, reduction="none")
    denom = m.sum(dim=0).clamp(min=1.0)
    head_loss = (bce * m).sum(dim=0) / denom
    return (head_loss * head_w).sum()


def _resolve_loss(policy: dict):
    """Deferred import breaks the train <-> losses import cycle."""
    from feed.losses import build_loss
    return build_loss(str(policy["train"].get("loss", "masked_bce")), policy)


def _val_ndcg(model: torch.nn.Module, val: dict, policy: dict, gains: list[int],
              k: int, keep: np.ndarray) -> float:
    probs = predict_probs(model, val["x"], val["cat"])
    scores = score_from_probs(probs, policy["score_weights"])
    ecfg = policy["eval"]
    _, nd = per_user_ndcg(scores, val["tier"], val["user_id"], gains, k, keep=keep,
                          min_impressions=ecfg["min_impressions"], min_tiers=ecfg["min_tiers"])
    return float(nd.mean()) if len(nd) else 0.0


@torch.no_grad()
def val_losses(model: torch.nn.Module, val: dict, pos_w: torch.Tensor, head_w: torch.Tensor,
               batch_size: int = 65536) -> tuple[float, list[float]]:
    """Masked per-head BCE on the val split (a common yardstick across all ablation arms)."""
    model.eval()
    xt = torch.from_numpy(val["x"])
    ct = torch.from_numpy(val["cat"].astype(np.int64))
    yt = torch.from_numpy(val["y"])
    mt = torch.from_numpy(val["m"])
    total, n = 0.0, len(xt)
    per_head = torch.zeros(len(HEADS))
    for i in range(0, n, batch_size):
        logits = model(xt[i:i + batch_size], ct[i:i + batch_size])
        loss = masked_head_loss(logits, yt[i:i + batch_size], mt[i:i + batch_size], pos_w, head_w)
        bce = F.binary_cross_entropy_with_logits(logits, yt[i:i + batch_size], pos_weight=pos_w,
                                                 reduction="none")
        ml = mt[i:i + batch_size]
        per_head += ((bce * ml).sum(dim=0) / ml.sum(dim=0).clamp(min=1.0)) * (len(logits) / n)
        total += float(loss) * (len(logits) / n)
    return total, [float(x) for x in per_head]


def _group_batches(uid: np.ndarray, batch_size: int, rng: np.random.Generator):
    """Yield row-index batches that keep each user's impressions whole (needed by listwise L3)."""
    order = np.argsort(uid, kind="stable")
    su = uid[order]
    starts = np.r_[0, np.flatnonzero(su[1:] != su[:-1]) + 1]
    groups = [order[a:b] for a, b in zip(starts, np.r_[starts[1:], len(order)])]
    perm = rng.permutation(len(groups))
    cur: list[np.ndarray] = []
    size = 0
    for g in perm:
        cur.append(groups[g])
        size += len(groups[g])
        if size >= batch_size:
            yield np.concatenate(cur)
            cur, size = [], 0
    if cur:
        yield np.concatenate(cur)


def train_taste(model: torch.nn.Module, train: dict, val: dict, policy: dict,
                gains: list[int], seed: int = 42, log=print) -> dict:
    """Train in-memory on the whole train split; returns history + best state info."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    tcfg, mcfg, ecfg = policy["train"], policy["model"], policy["eval"]
    head_w = torch.tensor([mcfg["loss_weights"][h] for h in HEADS], dtype=torch.float32)
    pos_w = torch.tensor(compute_pos_weights(train["y"], train["m"], mcfg["pos_weight_cap"],
                                             mcfg.get("pos_weight_mode", "sqrt")),
                         dtype=torch.float32)
    loss_fn, extra_params = _resolve_loss(policy)
    xt = torch.from_numpy(train["x"])
    ct = torch.from_numpy(train["cat"].astype(np.int64))
    yt = torch.from_numpy(train["y"])
    mt = torch.from_numpy(train["m"])
    uid_t = torch.from_numpy(train["user_id"])
    tier_t = torch.from_numpy(train["tier"].astype(np.int64))
    score_w = torch.tensor([policy["score_weights"][h] for h in HEADS], dtype=torch.float32)
    val_feed = (val["is_search"] == 0)
    n, bs = len(xt), int(tcfg["batch_size"])
    steps = int(np.ceil(n / bs))
    params = list(model.parameters()) + list(extra_params)
    opt = torch.optim.AdamW(params, lr=tcfg["lr"], weight_decay=tcfg["weight_decay"])
    if str(tcfg.get("scheduler", "onecycle")) == "cosine":
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=int(tcfg["max_epochs"]) * steps)
    else:
        sched = torch.optim.lr_scheduler.OneCycleLR(
            opt, max_lr=tcfg["lr"], total_steps=int(tcfg["max_epochs"]) * steps)
    clip = tcfg.get("grad_clip", None)
    grouped = bool(getattr(loss_fn, "needs_groups", False))
    history: list[dict] = []
    best = {"ndcg10": -1.0, "epoch": -1, "state": None}
    no_improve = 0
    for epoch in range(1, int(tcfg["max_epochs"]) + 1):
        model.train()
        t0 = time.time()
        running = 0.0
        per_head_run = torch.zeros(len(HEADS))
        batch_iter = (_group_batches(train["user_id"], bs, rng) if grouped
                      else (rng.permutation(n)[i:i + bs] for i in range(0, n, bs)))
        for idx in batch_iter:
            opt.zero_grad(set_to_none=True)
            out = loss_fn(model(xt[idx], ct[idx]), yt[idx], mt[idx], pos_w,
                          {"head_w": head_w, "score_w": score_w, "uid": uid_t[idx],
                           "tier": tier_t[idx], "gains": gains, "train": tcfg})
            out["total"].backward()
            if clip:
                torch.nn.utils.clip_grad_norm_(params, float(clip))
            opt.step()
            sched.step()
            running += float(out["total"].detach()) * len(idx)
            per_head_run += out["per_head"] * len(idx)
        v_loss, v_head = val_losses(model, val, pos_w, head_w)
        m10 = _val_ndcg(model, val, policy, gains, ecfg["ndcg_ks"][0], val_feed)
        m20 = _val_ndcg(model, val, policy, gains, ecfg["ndcg_ks"][1], val_feed)
        history.append({"epoch": epoch, "train_loss": running / n,
                        "train_per_head": [float(x) for x in per_head_run / n],
                        "val_loss": v_loss, "val_per_head": v_head,
                        "val_ndcg10": m10, "val_ndcg20": m20,
                        "lr": float(opt.param_groups[0]["lr"]), "seconds": time.time() - t0})
        log(f"[train] epoch {epoch}/{tcfg['max_epochs']} loss={running / n:.4f} "
            f"val_loss={v_loss:.4f} val ndcg@10={m10:.4f} ndcg@20={m20:.4f} "
            f"({time.time() - t0:.1f}s)")
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
            "n_train_rows": int(n),
            "loss": str(tcfg.get("loss", "masked_bce")),
            "pos_weight_mode": mcfg.get("pos_weight_mode", "sqrt")}
