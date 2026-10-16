"""Intuition: Find (build the 150-candidate retrieval pool) then Rank (one scorer per policy family),
with the pool held fixed across policies so the ladder isolates the ranker, not the retriever.
"""
from __future__ import annotations

import copy

import numpy as np
import torch

from feed.data import DENSE, load_scaler, standardize_matrix
from feed.model import build_model, load_meta, predict_probs, score_from_probs

_RETRIEVAL_MIX = ("n_popular", "n_affinity", "n_new", "n_random")


def effective_policy(policy: dict, name: str) -> dict:
    """Base policy, with the Phase-4 evolved knobs overlaid when the `evolved` rung is requested."""
    if name != "evolved":
        return policy
    over = policy.get("_evolved")
    if not over:
        raise ValueError("evolved policy requested but no _evolved overrides are loaded")
    p = copy.deepcopy(policy)
    for block in ("composer", "score_weights", "retrieval"):
        if block in over:
            p[block] = {**p[block], **over[block]}
    rc = p["retrieval"]
    rc["pool_size"] = int(sum(int(round(rc[k])) for k in _RETRIEVAL_MIX))
    return p


class TasteScorer:
    """Batched TasteNet inference on live raw features (standardised with the train scaler)."""

    def __init__(self, policy: dict, threads: int) -> None:
        torch.set_num_threads(int(threads))
        self.meta = load_meta()
        self.scaler = load_scaler()
        self.ckpt_name = policy.get("_taste_ckpt", "taste.pt")
        ckpt = torch.load(str(policy["_models_dir"] / self.ckpt_name), map_location="cpu",
                          weights_only=False)
        variant = policy.get("_model_variant") or ckpt.get("variant")
        self.model = build_model(self.meta, policy, variant=variant)
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.eval()
        self.variant = variant or "baseline"
        self.weights = policy["score_weights"]

    @torch.no_grad()
    def score(self, x: np.ndarray, cat: np.ndarray) -> np.ndarray:
        z = standardize_matrix(x, self.scaler)
        probs = predict_probs(self.model, z, cat)
        return score_from_probs(probs, self.weights).astype(np.float64)


def build_pool(world, fs, bots_idx: np.ndarray, ts: np.ndarray, rng: np.random.Generator,
               rc: dict) -> np.ndarray:
    """Concatenate popular + user-category affinity + cold/new + random candidates per session."""
    c = world.catalog
    n_sess = len(bots_idx)
    n_items = c.n_items
    pop = int(rc["n_popular"]); n_aff = int(rc["n_affinity"])
    n_new = int(rc["n_new"]); n_rand = int(rc["n_random"])
    score = fs.item_popularity()
    top = np.argpartition(-(score + rng.normal(0, 1e-9, n_items)), pop - 1)[:pop]
    parts = [np.tile(top.astype(np.int32), (n_sess, 1))]
    prior = float(rc.get("affinity_prior", 2.0))
    obs = fs.uc_n[bots_idx].astype(np.float64) + prior * c.cat_share
    obs /= np.maximum(obs.sum(axis=1, keepdims=True), 1e-9)
    cs = np.cumsum(obs, axis=1)
    cats = np.empty((n_sess, n_aff), np.int64)
    for j in range(n_aff):
        cats[:, j] = np.clip((rng.random(n_sess)[:, None] > cs).sum(axis=1), 0, c.n_cats - 1)
    sizes = c.cat_start[cats + 1] - c.cat_start[cats]
    off = (rng.random((n_sess, n_aff)) * sizes).astype(np.int64)
    parts.append(c.cat_items[c.cat_start[cats] + off].astype(np.int32))
    cold = np.flatnonzero(fs.cold_items(int(world.cfg["catalog"]["cold_max_n"])))
    if len(cold) == 0:
        cold = np.arange(n_items)
    parts.append(cold[rng.integers(0, len(cold), size=(n_sess, n_new))].astype(np.int32))
    parts.append(rng.integers(0, n_items, size=(n_sess, n_rand)).astype(np.int32))
    return np.concatenate(parts, axis=1)


def _zscore_rows(a: np.ndarray) -> np.ndarray:
    return (a - a.mean(axis=1, keepdims=True)) / np.maximum(a.std(axis=1, keepdims=True), 1e-6)


def rank_scores(name: str, world, fs, bots_idx: np.ndarray, pool: np.ndarray, ts: np.ndarray,
                rng: np.random.Generator, scorer: TasteScorer | None = None) -> np.ndarray:
    """(n_sessions, pool) policy scores; higher = shown earlier."""
    n_sess, P = pool.shape
    its = pool.reshape(-1).astype(np.int64)
    bs = np.repeat(bots_idx.astype(np.int64), P)
    tsf = np.repeat(np.asarray(ts, dtype=np.float64), P)
    if name == "random":
        return rng.random((n_sess, P))
    if name == "popularity":
        return fs.item_popularity()[its].reshape(n_sess, P)
    if name == "personal_affinity":
        rc = world.cfg["ranking"]
        pop = fs.item_popularity()[its].reshape(n_sess, P)
        exp_share, decay_share, _ = fs.uc_stats(bs, its, tsf)
        lift = decay_share / (exp_share + float(rc["lift_prior"]))
        lift = np.clip(lift, 0.0, float(rc["lift_cap"])).reshape(n_sess, P)
        return _zscore_rows(pop) + float(rc["affinity_weight"]) * _zscore_rows(lift)
    if name in ("taste", "taste_composer", "evolved"):
        x, cat = fs.pair_features(bs, its, tsf)
        return scorer.score(x, cat).reshape(n_sess, P)
    if name == "oracle":
        c = world.cfg["response"]
        match, q = world.appeal(bs, its)
        return (float(c["a_match"]) * (match - float(c["match_ref"]))
                + float(c["b_quality"]) * (q - 0.5)).reshape(n_sess, P)
    raise ValueError(f"unknown policy {name}")


def taste_with_prior(world, fs, bots_idx: np.ndarray, pool: np.ndarray, ts: np.ndarray,
                     scorer: TasteScorer) -> tuple[np.ndarray, np.ndarray]:
    """Taste scores for a pool plus each user's prior exposure count (feature `ui_prior_n`).

    One `pair_features` pass yields both, so the composer's seen-penalty is free.
    """
    n_sess, P = pool.shape
    its = pool.reshape(-1).astype(np.int64)
    bs = np.repeat(bots_idx.astype(np.int64), P)
    tsf = np.repeat(np.asarray(ts, dtype=np.float64), P)
    x, cat = fs.pair_features(bs, its, tsf)
    scores = scorer.score(x, cat).reshape(n_sess, P)
    prior = x[:, DENSE.index("ui_prior_n")].astype(np.float32).reshape(n_sess, P)
    return scores, prior


def order_from_scores(scores: np.ndarray, cap: int) -> np.ndarray:
    """Descending-rank item columns per session, truncated to the session cap."""
    order = np.argsort(-scores, axis=1, kind="stable")[:, :cap]
    return order.astype(np.int32)
