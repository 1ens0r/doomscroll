"""Intuition: One in-process train+evaluate routine shared by the capacity and loss ablations, so
every arm runs through the identical data, split, candidate protocol, metric, and seed -- the only
thing that changes is the config."""
from __future__ import annotations

import copy
import resource
import time

import numpy as np
import torch

from feed.data import load_scaler, load_split, load_xy
from feed.eval import baseline_scores, bootstrap_ci, paired_bootstrap, per_user_ndcg
from feed.model import build_model, n_params, predict_probs, score_from_probs
from feed.train import train_taste

FEED_SCOPES = ("feed", "feed_cold")


def peak_rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def load_bundles(policy: dict, max_rows: int | None = None, seed: int = 42) -> dict:
    """Train-only scaler; val/test standardised with it. Optional row cap for smoke runs."""
    scaler = load_scaler()
    rng = np.random.default_rng(seed)
    out: dict = {}
    for split in ("train", "val", "test"):
        b = load_xy(split, scaler)
        raw = load_split(split, columns=["i_mean_score", "uc_lift"])
        b["quality"] = raw["i_mean_score"].to_numpy(np.float64)
        b["lift"] = raw["uc_lift"].to_numpy(np.float64)
        if max_rows is not None and len(b["x"]) > max_rows:
            idx = np.sort(rng.choice(len(b["x"]), size=max_rows, replace=False))
            b = {k: v[idx] for k, v in b.items()}
        out[split] = b
    return out


def _scope_keep(bundle: dict, scope: str) -> np.ndarray:
    feed = bundle["is_search"] == 0
    return feed & (bundle["cold_item"] == 1) if scope == "feed_cold" else feed


def per_user_scores(score: np.ndarray, bundle: dict, policy: dict, gains: list[int],
                    k: int, scope: str = "feed") -> tuple[np.ndarray, np.ndarray]:
    ecfg = policy["eval"]
    return per_user_ndcg(score, bundle["tier"], bundle["user_id"], gains, k,
                         keep=_scope_keep(bundle, scope),
                         min_impressions=ecfg["min_impressions"], min_tiers=ecfg["min_tiers"])


def summarize(users: np.ndarray, vals: np.ndarray, policy: dict) -> dict:
    return bootstrap_ci(vals, policy["eval"]["bootstrap_resamples"], policy["eval"]["bootstrap_seed"])


def run_one(bundles: dict, policy: dict, meta: dict, gains: list[int], name: str,
            variant: str | None = None, evaluate_test: bool = False, seed: int = 42,
            ckpt_dir=None, return_model: bool = False, log=print) -> dict:
    """Train one arm and evaluate it on the feed scope; test only when explicitly asked."""
    pol = copy.deepcopy(policy)
    pol["_gains"] = list(gains)
    model = build_model(meta, pol, variant=variant)
    params = n_params(model)
    log(f"[{name}] variant={variant or 'baseline'} loss={pol['train'].get('loss', 'masked_bce')} "
        f"params={params:,}")
    t0 = time.time()
    report = train_taste(model, bundles["train"], bundles["val"], pol, gains, seed, log=log)
    train_seconds = time.time() - t0
    ckpt_path = None
    if ckpt_dir is not None:
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        ckpt_path = ckpt_dir / f"{name}.pt"
        torch.save({"state_dict": model.state_dict(), "variant": variant, "name": name,
                    "model_cfg": pol["model"], "history": report["history"]}, ckpt_path)
    result: dict = {"name": name, "variant": variant or "baseline",
                    "loss": pol["train"].get("loss", "masked_bce"), "n_params": params,
                    "best_epoch": report["best_epoch"], "best_val_ndcg10": report["best_val_ndcg10"],
                    "pos_weights": report["pos_weights"], "history": report["history"],
                    "pos_weight_mode": report.get("pos_weight_mode", "sqrt"),
                    "train_seconds": train_seconds, "peak_rss_mb": peak_rss_mb(),
                    "ckpt": str(ckpt_path) if ckpt_path else None,
                    "scopes": {}}
    for split in (("val", "test") if evaluate_test else ("val",)):
        bundle = bundles[split]
        probs = predict_probs(model, bundle["x"], bundle["cat"])
        scores = {"taste": score_from_probs(probs, pol["score_weights"])}
        scores.update(baseline_scores(bundle["quality"], bundle["lift"], len(bundle["x"]),
                                      pol["seed"]))
        split_out: dict = {}
        for scope in FEED_SCOPES:
            scope_out: dict = {}
            for method, sc in scores.items():
                entry = {"per_user": {}}
                for k in pol["eval"]["ndcg_ks"]:
                    users, vals = per_user_scores(sc, bundle, pol, gains, k, scope)
                    entry["per_user"][str(k)] = (users, vals)
                    entry[f"ndcg@{k}"] = summarize(users if len(users) else vals, vals, pol)
                scope_out[method] = entry
            split_out[scope] = scope_out
        result["scopes"][split] = split_out
    if return_model:
        result["model"] = model
        result["policy"] = pol
    return result


def paired_vs(result: dict, other: dict, scope: str = "feed", split: str = "val",
              k: int = 10, policy: dict | None = None) -> dict:
    """Paired user-bootstrap CI of mean(other - result) on the same users for one split/scope."""
    a = result["scopes"][split][scope]["taste"]["per_user"][str(k)]
    b = other["scopes"][split][scope]["taste"]["per_user"][str(k)]
    if not np.array_equal(a[0], b[0]):
        raise RuntimeError(f"paired user mismatch {result['name']} vs {other['name']}")
    n_boot = 2000 if policy is None else policy["eval"]["bootstrap_resamples"]
    seed = 42 if policy is None else policy["eval"]["bootstrap_seed"]
    return paired_bootstrap(b[1], a[1], n_boot, seed)


def strip(result: dict) -> dict:
    """JSON-able view: drop the per-user arrays."""
    out = {k: v for k, v in result.items() if k not in ("scopes",)}
    out["scopes"] = {
        split: {scope: {m: {k: v for k, v in e.items() if k != "per_user"}
                        for m, e in methods.items()}
                for scope, methods in scopes.items()}
        for split, scopes in result["scopes"].items()}
    return out
