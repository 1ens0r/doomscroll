"""Intuition: fast tests for the Phase-8 loss framework: L0 parity with the shipped objective, mask
invariance, finite/backprop-able objectives, weighting modes, and calibration diagnostics."""
import numpy as np
import torch

from feed.losses import (LambdaRankLoss, UncertaintyWeighting, brier, build_loss, ece,
                         equal_head_bce, log_loss, masked_bce, reliability)
from feed.train import compute_pos_weights, masked_head_loss

GAINS = [0, 1, 2, 3, 4]
HEADS = ["click", "like", "follow", "forward", "watch"]


def _batch(n: int = 64, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    logits = torch.randn(n, 5, generator=g, requires_grad=True)
    y = (torch.rand(n, 5, generator=g) < 0.2).float()
    m = torch.ones(n, 5)
    m[n // 2:, 2:] = 0.0
    pos_w = torch.tensor([1.0, 2.0, 3.0, 3.0, 1.1])
    head_w = torch.tensor([1.0, 2.0, 3.0, 3.0, 1.0])
    return logits, y, m, pos_w, head_w


def test_l0_parity_with_shipped_objective():
    logits, y, m, pos_w, head_w = _batch()
    ref = masked_head_loss(logits, y, m, pos_w, head_w)
    got = masked_bce(logits, y, m, pos_w, {"head_w": head_w})["total"]
    assert torch.allclose(ref, got)


def test_mask_invariance_of_masked_loss():
    logits, y, m, pos_w, head_w = _batch()
    y2 = y.clone()
    y2[m == 0] = 1.0 - y2[m == 0]  # flip only unobserved labels
    a = masked_bce(logits, y, m, pos_w, {"head_w": head_w})["total"]
    b = masked_bce(logits, y2, m, pos_w, {"head_w": head_w})["total"]
    assert torch.allclose(a, b)


def test_all_losses_finite_and_backprop():
    uv = UncertaintyWeighting(5)
    lr = LambdaRankLoss(GAINS)
    fns = [masked_bce, equal_head_bce, uv,
           lambda lg, yy, mm, pw, ctx: lr(lg, yy, mm, pw, ctx)]
    for fn in fns:
        logits, y, m, pos_w, head_w = _batch()
        uid = torch.randint(0, 8, (64,))
        tier = torch.randint(0, 5, (64,))
        out = fn(logits, y, m, pos_w, {"head_w": head_w, "score_w": torch.ones(5),
                                       "uid": uid, "tier": tier, "gains": GAINS})
        assert torch.isfinite(out["total"]) and len(out["per_head"]) == 5
        out["total"].backward()
        assert logits.grad is not None and torch.isfinite(logits.grad).all()


def test_build_loss_registry_and_extra_params():
    _, p0 = build_loss("masked_bce", {"train": {}})
    assert p0 == []
    fn, p1 = build_loss("uncertainty", {"train": {}})
    assert len(p1) == 1 and p1[0].requires_grad
    fn2, p2 = build_loss("lambdarank", {"_gains": GAINS, "train": {}})
    assert p2 == [] and getattr(fn2, "needs_groups", False)
    fn3, p3 = build_loss("hybrid", {"_gains": GAINS, "train": {"rank_weight": 0.5}})
    assert p3 == [] and getattr(fn3, "needs_groups", False)
    logits, y, m, pos_w, head_w = _batch()
    out = fn3(logits, y, m, pos_w, {"head_w": head_w, "score_w": torch.ones(5),
                                    "uid": torch.randint(0, 8, (64,)),
                                    "tier": torch.randint(0, 5, (64,)), "gains": GAINS})
    assert torch.isfinite(out["total"])
    try:
        build_loss("nope", {"train": {}})
    except ValueError:
        return
    raise AssertionError("expected ValueError for unknown loss")


def test_pos_weight_modes():
    y = np.zeros((101, 1))
    y[0, 0] = 1.0
    m = np.ones_like(y)
    assert compute_pos_weights(y, m, cap=10.0, mode="none")[0] == 1.0
    assert compute_pos_weights(y, m, cap=10.0, mode="sqrt")[0] == 10.0
    enn = compute_pos_weights(y, m, cap=10.0, mode="enn")[0]
    assert 0.0 < enn <= 10.0
    balanced = np.array([[1.0], [0.0], [1.0], [0.0]])
    assert np.isclose(compute_pos_weights(balanced, np.ones_like(balanced), 10.0)[0], 1.0)


def test_calibration_diagnostics_toy():
    p = np.array([0.1, 0.2, 0.8, 0.9])
    y = np.array([0.0, 0.0, 1.0, 1.0])
    assert 0.0 <= ece(p, y) < 0.2
    assert 0.0 <= brier(p, y) < 0.1
    assert log_loss(p, y) < 0.5
    rel = reliability(p, y)
    assert len(rel["counts"]) == 10 and rel["n"] == 4
    masked = reliability(np.array([0.9, 0.9]), np.array([1.0, 0.0]), np.array([1.0, 0.0]))
    assert masked["n"] == 1
