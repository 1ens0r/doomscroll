"""Intuition: Find (build the 150-candidate retrieval pool) then Rank (one scorer per policy family),
with the pool held fixed across policies so the ladder isolates the ranker, not the retriever.
"""
from __future__ import annotations

import numpy as np
import torch

from feed.data import load_scaler, standardize_matrix
from feed.model import build_model, load_meta, predict_probs, score_from_probs


class TasteScorer:
    """Batched TasteNet inference on live raw features (standardised with the train scaler)."""

    def __init__(self, policy: dict, threads: int) -> None:
        torch.set_num_threads(int(threads))
        self.meta = load_meta()
        self.scaler = load_scaler()
        self.model = build_model(self.meta, policy)
        ckpt = torch.load(str(policy["_models_dir"] / "taste.pt"), map_location="cpu",
                          weights_only=False)
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.eval()
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
    p = world.bots.taste[bots_idx].astype(np.float64)
    cs = np.cumsum(p, axis=1)
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
    if name in ("taste", "taste_composer"):
        x, cat = fs.pair_features(bs, its, tsf)
        return scorer.score(x, cat).reshape(n_sess, P)
    if name == "oracle":
        c = world.cfg["response"]
        cat = world.catalog.cat[its]
        match = world.bots.taste[bs, cat].astype(np.float64)
        q = world.catalog.q[its].astype(np.float64)
        return (float(c["a_match"]) * (match - float(c["match_ref"]))
                + float(c["b_quality"]) * (q - 0.5)).reshape(n_sess, P)
    raise ValueError(f"unknown policy {name}")


def order_from_scores(scores: np.ndarray, cap: int) -> np.ndarray:
    """Descending-rank item columns per session, truncated to the session cap."""
    order = np.argsort(-scores, axis=1, kind="stable")[:, :cap]
    return order.astype(np.int32)
