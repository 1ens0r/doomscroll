"""Intuition: fast tests for the Phase-7 capacity machinery: variants resolve and instantiate,
the baseline stays byte-for-byte the original architecture, the cross term only appears when asked,
and every variant produces finite logits of the right shape and trains end to end."""
import numpy as np
import torch

from feed.data import DENSE
from feed.model import TasteNet, block_indices, build_model, list_variants, n_params, resolve_model_cfg

POLICY = {
    "model": {
        "dropout": 0.1, "user_hidden": [64, 32], "video_hidden": [64, 32], "pair_hidden": [32],
        "fusion_hidden": [128, 64], "cat_emb_dim": 16, "activation": "relu", "norm": "none",
        "cross_rank": 0,
        "loss_weights": {"click": 1.0, "like": 2.0, "follow": 3.0, "forward": 3.0, "watch": 1.0},
        "pos_weight_cap": 10.0, "pos_weight_mode": "sqrt",
        "variants": [
            {"name": "wide", "user_hidden": [128, 64], "video_hidden": [128, 64],
             "fusion_hidden": [256, 128]},
            {"name": "cross", "cross_rank": 32},
            {"name": "gelu_norm", "activation": "gelu", "norm": "layernorm"},
        ],
    }
}
META = {"dense": DENSE, "cat_embed_rows": 38}


def _dummy(n: int = 32) -> tuple[torch.Tensor, torch.Tensor]:
    return torch.randn(n, len(DENSE)), torch.randint(0, 38, (n,))


def test_baseline_resolution_is_base_config():
    base = resolve_model_cfg(POLICY, None)
    assert resolve_model_cfg(POLICY, "baseline") == base
    assert "variants" not in base
    assert base["user_hidden"] == [64, 32]


def test_baseline_param_count_matches_shipped_model():
    model = build_model(META, POLICY, variant="baseline")
    assert n_params(model) == 32357


def test_variants_instantiate_and_grow():
    assert "wide" in list_variants(POLICY)
    base = n_params(build_model(META, POLICY, "baseline"))
    wide = n_params(build_model(META, POLICY, "wide"))
    assert wide > base
    assert hasattr(build_model(META, POLICY, "cross"), "cross_u")
    assert not hasattr(build_model(META, POLICY, "baseline"), "cross_u")


def test_forward_shapes_and_gradients_all_variants():
    for variant in ("baseline", "wide", "cross", "gelu_norm"):
        model = build_model(META, POLICY, variant=variant)
        x, c = _dummy()
        out = model(x, c)
        assert out.shape == (32, 5) and torch.isfinite(out).all()
        out.sum().backward()
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        assert grads and all(torch.isfinite(g).all() for g in grads)


def test_unknown_variant_fails_loudly():
    try:
        resolve_model_cfg(POLICY, "does_not_exist")
    except ValueError:
        return
    raise AssertionError("expected ValueError for unknown variant")


def test_cross_term_changes_output_vs_zero_rank():
    a = build_model(META, POLICY, "baseline")
    b = build_model(META, POLICY, "cross")
    assert n_params(b) > n_params(a)
    x, c = _dummy(8)
    assert not np.allclose(a(x, c).detach().numpy(), b(x, c).detach().numpy())
