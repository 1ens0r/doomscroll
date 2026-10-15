"""Intuition: The simulated world — a catalog of items with hidden appeal plus live visible stats,
and a population of bots with hidden taste, activity rhythm and session psychology.

Everything here is seeded from the real model population (`item_stats`, `user_stats`,
`user_interest`, `session_hours`) so the swarm speaks in the same units as the offline features.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from feed.content import item_idiosyncrasy, preference_blocks

ROOT = Path(__file__).resolve().parent.parent
ART = ROOT / "artifacts"
SWARM_PATH = ROOT / "config" / "swarm.yaml"
CALIB_PATH = ART / "calibration.json"
DAY_MS = 86_400_000.0
HOUR_MS = 3_600_000.0
MIN_MS = 60_000.0
N_CATS = 38


def load_config(use_calibration: bool = True) -> dict:
    """swarm.yaml, then layer fitted constants from artifacts/calibration.json on top."""
    with open(SWARM_PATH) as f:
        cfg = yaml.safe_load(f)
    cfg["_calibrated"] = False
    if use_calibration and CALIB_PATH.exists():
        with open(CALIB_PATH) as f:
            calib = json.load(f)
        for block in ("response", "leave", "return_model"):
            for k, v in calib.get(block, {}).items():
                if k in cfg.get(block, {}):
                    cfg[block][k] = v
        cfg["_calibrated"] = True
    return cfg


def _sample_indices(n: int, size: int, weights: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    if weights is not None and weights.size == n and size < n:
        return rng.choice(n, size=size, replace=False, p=weights)
    return rng.choice(n, size=size, replace=size > n)


@dataclass
class Catalog:
    """Static items + hidden appeal; `init_*` seeds the live visible counters in FeatureState."""

    n_items: int
    n_cats: int
    item_id: np.ndarray
    cat: np.ndarray
    q: np.ndarray
    idio: np.ndarray
    duration: np.ndarray
    init_n: np.ndarray
    init_sum_score: np.ndarray
    init_sum_pos: np.ndarray
    init_sum_play: np.ndarray
    init_n_play: np.ndarray
    pop_sum: np.ndarray
    pop_count: np.ndarray
    cat_start: np.ndarray = field(default=None)
    cat_items: np.ndarray = field(default=None)
    cat_share: np.ndarray = field(default=None)
    _raw: pd.DataFrame = field(default=None, repr=False)

    def reindex_cats(self) -> None:
        order = np.argsort(self.cat, kind="stable").astype(np.int64)
        self.cat_items = order
        counts = np.bincount(self.cat, minlength=self.n_cats)
        self.cat_start = np.r_[0, np.cumsum(counts)].astype(np.int64)
        self.cat_share = (counts / max(counts.sum(), 1)).astype(np.float64)

    def daily_replace(self, frac: float, day: int, rng: np.random.Generator) -> np.ndarray:
        """Swap `frac` of the catalog for fresh zero-stat items (cold-start pressure)."""
        k = max(1, int(round(frac * self.n_items)))
        idx = rng.choice(self.n_items, size=k, replace=False)
        raw = self._raw
        pick = rng.integers(0, len(raw), size=k)
        self.cat[idx] = raw["cat_id"].to_numpy(np.int8)[pick]
        base = np.clip(raw["mean_score"].to_numpy(np.float64)[pick] / self._q_scale, 0, 1)
        self.q[idx] = np.clip(base + rng.normal(0, self._q_noise, k), self._q_floor, 1.0)
        self.idio[idx] = item_idiosyncrasy(k, self.idio.shape[1], rng)
        self.duration[idx] = raw["duration_ms"].to_numpy(np.float32)[pick]
        self.item_id[idx] = -(day * 1_000_000 + np.arange(k) + 1)
        for arr in (self.init_n, self.init_sum_score, self.init_sum_pos,
                    self.init_sum_play, self.init_n_play, self.pop_sum, self.pop_count):
            arr[idx] = 0.0
        self.reindex_cats()
        return idx


def build_catalog(cfg: dict, seed: int, init_stats: bool = True) -> Catalog:
    """Sample the catalog by exposure^power so the popularity head and tail both exist."""
    rng = np.random.default_rng(np.random.SeedSequence([seed, 11]))
    items = pd.read_parquet(ART / "item_stats.parquet",
                            columns=["item_id", "cat_id", "n", "mean_score", "pos_rate",
                                     "play_ratio", "duration_ms"])
    ccfg = cfg["catalog"]
    w = np.power(items["n"].to_numpy(np.float64), float(ccfg["sample_power"]))
    w = w / w.sum()
    idx = _sample_indices(len(items), int(ccfg["n_items"]), w, rng)
    cat = items["cat_id"].to_numpy(np.int8)[idx]
    n = items["n"].to_numpy(np.float64)[idx]
    base = np.clip(items["mean_score"].to_numpy(np.float64)[idx] / float(ccfg["q_scale"]), 0, 1)
    q = np.clip(base + rng.normal(0, float(ccfg["q_noise"]), len(idx)), float(ccfg["q_floor"]), 1.0)
    idio = item_idiosyncrasy(len(idx), int(ccfg["idio_dim"]), rng)
    if init_stats:
        init_n, init_ss = n, items["mean_score"].to_numpy(np.float64)[idx] * n
        init_sp = items["pos_rate"].to_numpy(np.float64)[idx] * n
        init_splay = items["play_ratio"].to_numpy(np.float64)[idx] * n
        init_nplay = n.copy()
    else:
        init_n = np.zeros(len(idx)); init_ss = init_n.copy(); init_sp = init_n.copy()
        init_splay = init_n.copy(); init_nplay = init_n.copy()
    prior = float(ccfg.get("pop_prior_count", 20.0))
    gmean = float(init_ss.sum() / max(init_n.sum(), 1.0))
    pop_sum = init_ss + prior * gmean
    pop_count = init_n + prior
    cat_obj = Catalog(
        n_items=len(idx), n_cats=N_CATS,
        item_id=items["item_id"].to_numpy(np.int32)[idx], cat=cat, q=q.astype(np.float32),
        idio=idio, duration=items["duration_ms"].to_numpy(np.float32)[idx],
        init_n=init_n, init_sum_score=init_ss, init_sum_pos=init_sp,
        init_sum_play=init_splay, init_n_play=init_nplay,
        pop_sum=pop_sum, pop_count=pop_count)
    cat_obj.reindex_cats()
    cat_obj._raw = items
    cat_obj._q_scale, cat_obj._q_noise, cat_obj._q_floor = (
        float(ccfg["q_scale"]), float(ccfg["q_noise"]), float(ccfg["q_floor"]))
    return cat_obj


@dataclass
class Bots:
    """Hidden taste + activity + a pre-drawn, policy-independent session schedule (CRN)."""

    n_bots: int
    n_cats: int
    user_id: np.ndarray
    taste: np.ndarray
    breadth: np.ndarray
    sessions_per_day: np.ndarray
    like_mult: np.ndarray
    follow_mult: np.ndarray
    forward_mult: np.ndarray
    counts: np.ndarray
    hour: np.ndarray
    patience: np.ndarray
    return_u: np.ndarray
    pref_cat: np.ndarray = field(default=None)
    pref_idio: np.ndarray = field(default=None)


def build_bots(cfg: dict, seed: int, n_bots: int | None = None,
               n_days: int | None = None) -> Bots:
    """Stratify bots by activity tercile; hidden taste is the real category decay-share."""
    rng = np.random.default_rng(np.random.SeedSequence([seed, 12]))
    us = pd.read_parquet(ART / "user_stats.parquet")
    ui = pd.read_parquet(ART / "user_interest.parquet")
    n_bots = int(n_bots or cfg["swarm"]["bots"])
    n_days = int(n_days or cfg["swarm"]["days"])
    waves = int(cfg["swarm"]["waves_per_day"])
    uid = us["user_id"].to_numpy(np.int64)
    pos = {int(u): i for i, u in enumerate(uid)}
    taste_all = np.zeros((len(uid), N_CATS), dtype=np.float64)
    share = np.nan_to_num(ui["decay_share"].to_numpy(np.float64), nan=0.0)
    taste_all[ui["user_id"].map(pos).to_numpy(), ui["cat_id"].to_numpy(np.int64)] = share
    activity = us["n_events"].to_numpy(np.float64)
    terciles = np.array_split(np.argsort(activity, kind="stable"), 3)
    per = n_bots // 3
    picks = np.concatenate([rng.choice(t, size=per, replace=True) for t in terciles]
                           + [rng.choice(len(uid), size=n_bots - per * 3, replace=True)])
    rng.shuffle(picks)
    taste = np.maximum(taste_all[picks], 1e-4)
    taste /= taste.sum(axis=1, keepdims=True)
    breadth = -(taste * np.log(taste)).sum(axis=1)
    spd = np.clip(us["sessions_per_day"].to_numpy(np.float64)[picks], 0.05, waves)
    def _mult(col: str) -> np.ndarray:
        v = us[col].to_numpy(np.float64)[picks]
        return np.clip(v / max(v.mean(), 1e-6), 0.2, 6.0)
    with open(ART / "session_hours.json") as f:
        hours = json.load(f)
    hp = np.array([hours[str(h)] for h in range(24)], dtype=np.float64)
    hp /= hp.sum()
    lcfg = cfg["leave"]
    counts = np.zeros((n_bots, n_days), np.int16)
    hour = np.zeros((n_bots, n_days, waves), np.int8)
    patience = np.zeros((n_bots, n_days, waves), np.float32)
    return_u = np.zeros((n_bots, n_days), np.float32)
    for i, child in enumerate(np.random.SeedSequence([seed, 13]).spawn(n_bots)):
        r = np.random.default_rng(child)
        counts[i] = np.minimum(r.poisson(spd[i], size=n_days), waves)
        hour[i] = r.choice(24, size=(n_days, waves), p=hp).astype(np.int8)
        patience[i] = np.exp(r.normal(np.log(float(lcfg["patience_median"])),
                                       float(lcfg["patience_sigma"]), size=(n_days, waves)))
        return_u[i] = r.random(n_days).astype(np.float32)
    return Bots(n_bots, N_CATS, uid[picks], taste.astype(np.float32), breadth.astype(np.float32),
                spd, _mult("rate_like"), _mult("rate_follow"), _mult("rate_forward"),
                counts, hour, patience, return_u)


@dataclass
class World:
    """Catalog + bots + seed. `rng(*key)` gives reproducible, position-independent streams."""

    catalog: Catalog
    bots: Bots
    cfg: dict
    seed: int

    def rng(self, *key: int) -> np.random.Generator:
        return np.random.default_rng(np.random.SeedSequence([self.seed, *key]))

    def appeal(self, bots_idx: np.ndarray, items: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Hidden user-conditional match and global quality for (bot, item) pairs.

        `match` = the bot's standardised category affinity for the item's category plus a small
        within-category latent term, so it is mean-zero per bot and `match_ref = 0` is neutral.
        """
        c = self.catalog
        cat_match = self.bots.pref_cat[bots_idx, c.cat[items]]
        idio = np.einsum("nk,nk->n", self.bots.pref_idio[bots_idx].astype(np.float64),
                         c.idio[items].astype(np.float64))
        gain = float(self.cfg["catalog"]["idio_gain"])
        scale = float(self.cfg["catalog"]["match_scale"])
        return scale * (cat_match.astype(np.float64) + gain * idio), c.q[items].astype(np.float64)


def build_world(cfg: dict, seed: int, n_bots: int | None = None, n_days: int | None = None,
                init_stats: bool = True) -> World:
    catalog = build_catalog(cfg, seed, init_stats)
    bots = build_bots(cfg, seed, n_bots, n_days)
    rng = np.random.default_rng(np.random.SeedSequence([seed, 14]))
    bots.pref_cat, bots.pref_idio = preference_blocks(
        bots.taste.astype(np.float64), int(cfg["catalog"]["idio_dim"]), rng)
    return World(catalog, bots, cfg, seed)
