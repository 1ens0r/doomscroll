"""Intuition: The fast-thinking Compose stage — a cheap numpy re-ranking applied on every swipe,
punishing repeats and category runs, rewarding session-level affinity and under-exposed items.
"""
from __future__ import annotations

import numpy as np

KNOWS = ("gamma", "alpha", "eta", "beta", "epsilon", "decay", "window_m")


def zscore_rows(scores: np.ndarray) -> np.ndarray:
    m = scores.mean(axis=1, keepdims=True)
    s = np.maximum(scores.std(axis=1, keepdims=True), 1e-6)
    return ((scores - m) / s).astype(np.float32)


class ComposerState:
    """Per-wave composer memory: what each session has already seen and how its taste is adapting."""

    def __init__(self, taste_z: np.ndarray, pool_cat: np.ndarray, pool_i_n: np.ndarray,
                 prior_seen: np.ndarray, cold_mask: np.ndarray, n_cats: int, knobs: dict) -> None:
        self.taste_z = taste_z.astype(np.float32)
        self.pool_cat = pool_cat.astype(np.int64)
        self.inv_i = (1.0 / np.sqrt(1.0 + pool_i_n.astype(np.float32)))
        self.shown = np.zeros_like(pool_i_n, dtype=np.int16)
        self.prior = (np.zeros_like(pool_i_n, dtype=np.float32) if prior_seen is None
                      else prior_seen.astype(np.float32))
        self.cold = cold_mask
        n_sess = taste_z.shape[0]
        m = int(knobs["window_m"])
        self.recent = np.full((n_sess, m), -1, dtype=np.int64)
        self.recent_n = np.zeros(n_sess, dtype=np.int64)
        self.aff = np.zeros((n_sess, n_cats), dtype=np.float32)
        self.k = {key: float(knobs[key]) for key in ("gamma", "alpha", "eta", "beta")}
        self.epsilon = float(knobs["epsilon"])
        self.decay = float(knobs["decay"])
        self.m = max(1, m)

    def _same_cat(self, act: np.ndarray) -> np.ndarray:
        cats = self.pool_cat[act]
        rec = self.recent[act]
        hit = cats[:, :, None] == rec[:, None, :]
        return hit.sum(axis=2).astype(np.float32)

    def select(self, act: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """Argmax of the composer score over the pool, with an epsilon exploration slot."""
        z = self.taste_z[act]
        aff = self.aff[act][np.arange(len(act))[:, None], self.pool_cat[act]]
        seen = self.prior[act] + self.shown[act]
        score = (z - self.k["gamma"] * seen - self.k["alpha"] * self._same_cat(act)
                 + self.k["eta"] * aff + self.k["beta"] * self.inv_i[act])
        score = np.where(self.shown[act] > 0, -1e9, score)
        best = np.argmax(score, axis=1)
        if self.epsilon > 0:
            explore = (rng.random(len(act)) < self.epsilon) & self.cold[act].any(axis=1)
            cold_score = np.where(self.cold[act] & (self.shown[act] == 0), score, -1e9)
            best = np.where(explore, np.argmax(cold_score, axis=1), best)
        return best.astype(np.int32)

    def observe(self, act: np.ndarray, pick: np.ndarray, reward: np.ndarray) -> None:
        """Record the chosen item: count it seen, push its category, update session affinity."""
        rows = np.arange(len(act))
        self.shown[act, pick] += 1
        cats = self.pool_cat[act, pick]
        idx = self.recent_n[act] % self.m
        self.recent[act, idx] = cats
        self.recent_n[act] += 1
        self.aff[act, cats] = self.decay * self.aff[act, cats] + reward
