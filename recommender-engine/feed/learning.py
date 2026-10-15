"""Intuition: The Learn stage — run the simulator, record the exact (features, actions) the live
serving path produced, and hand the ranker a training set drawn from its own deployment
distribution. This closes the offline/online gap that made the offline-trained model mis-rank.
"""
from __future__ import annotations

import numpy as np

from feed.model import HEADS


def generate_logs(cfg: dict, policy_name: str, seed: int, n_bots: int, n_days: int,
                  log=print) -> dict:
    """Collect one row per impression the live loop serves, under `policy_name` behaviour."""
    from feed.swarm import simulate
    store: dict[str, list] = {k: [] for k in ("x", "cat", "y", "user", "tier")}

    def collector(x: np.ndarray, cat: np.ndarray, out: dict, b: np.ndarray) -> None:
        store["x"].append(np.asarray(x, dtype=np.float32))
        store["cat"].append(np.asarray(cat, dtype=np.int64))
        store["y"].append(np.column_stack([np.asarray(out[h], dtype=np.float32) for h in HEADS]))
        store["user"].append(np.asarray(b, dtype=np.int64))
        store["tier"].append(np.asarray(out["tier"], dtype=np.int8))

    simulate(policy_name, cfg, seed, None, n_bots=n_bots, n_days=n_days,
             collector=collector, log=log)
    raw = {k: np.concatenate(v) for k, v in store.items()}
    log(f"[learn] collected {len(raw['x']):,} impressions from '{policy_name}' "
        f"({n_bots} bots x {n_days}d)")
    return raw


def make_bundle(raw: dict, scaler: dict, val_frac: float, seed: int, log=print) -> tuple[dict, dict]:
    """Standardise with the train-only scaler and split by bot so val users are unseen."""
    from feed.data import standardize_matrix
    users = np.unique(raw["user"])
    rng = np.random.default_rng(seed)
    val_users = rng.choice(users, size=max(1, int(val_frac * len(users))), replace=False)
    is_val = np.isin(raw["user"], val_users)

    def part(mask: np.ndarray) -> dict:
        n = int(mask.sum())
        return {
            "x": standardize_matrix(raw["x"][mask], scaler), "cat": raw["cat"][mask],
            "y": raw["y"][mask], "m": np.ones((n, len(HEADS)), dtype=np.float32),
            "user_id": raw["user"][mask], "tier": raw["tier"][mask],
            "is_search": np.zeros(n, dtype=np.int8), "cold_item": np.zeros(n, dtype=np.int8),
        }

    train, val = part(~is_val), part(is_val)
    log(f"[learn] split by bot: train={len(train['x']):,} rows / val={len(val['x']):,} rows "
        f"({len(users) - len(val_users)} / {len(val_users)} bots)")
    return train, val
