"""Intuition: The lockstep swarm — all active sessions advance one swipe at a time through
Find -> Rank -> Compose -> response -> feedback, with shared random numbers so policies are
compared on identical bots, schedules and draws. Returns IWE plus every diagnostic the plan needs.
"""
from __future__ import annotations

import time

import numpy as np

from feed.composer import ComposerState, zscore_rows
from feed.featurestate import FeatureState, hour_sincos  # noqa: F401
from feed.metrics import Metrics
from feed.policies import (TasteScorer, build_pool, order_from_scores, rank_scores,
                           taste_with_prior)
from feed.response import ResponseModel, leave_step, return_prob
from feed.world import DAY_MS, HOUR_MS, World, build_world

POLICIES = ["random", "popularity", "personal_affinity", "taste", "taste_composer", "oracle"]


def run_policy(world: World, policy: str, scorer: TasteScorer | None = None,
               collect_metrics: bool = True, collector=None, log=print) -> dict:
    cfg = world.cfg
    bots, cat = world.bots, world.catalog
    n_bots, n_days = bots.n_bots, bots.counts.shape[1]
    waves, cap = int(cfg["swarm"]["waves_per_day"]), int(cfg["swarm"]["max_session_items"])
    cap = min(cap, int(cfg["policy"]["retrieval"]["pool_size"]))
    fs = FeatureState(cat, n_bots)
    resp = ResponseModel(cfg)
    met = Metrics(n_bots, cat.n_items, cat.n_cats, cfg["iwe"], int(cfg["response"]["satiation_window"]))
    cold_max = int(cfg["catalog"]["cold_max_n"])
    active = np.ones(n_bots, dtype=bool)
    t0 = time.time()
    for d in range(n_days):
        if d > 0 and float(cfg["catalog"]["daily_replacement"]) > 0:
            repl = cat.daily_replace(float(cfg["catalog"]["daily_replacement"]), d,
                                     world.rng(5, d))
            fs.reset_items(repl)
        met.end_day(active)
        for w in range(waves):
            sess_bots = np.flatnonzero(active & (bots.counts[:, d] > w))
            if len(sess_bots) == 0:
                continue
            _run_wave(world, fs, resp, met, policy, scorer, sess_bots, d, w, cap, cold_max,
                      collect_metrics, collector)
        sat, skip, newf, has = met.day_stats()
        sat = np.where(has, sat, float(cfg["return_model"]["satisfaction_baseline"]))
        prob = return_prob(sat, newf, skip, cfg["return_model"])
        if d < n_days - 1:
            active = active & (bots.return_u[:, d] < prob)
        met.reset_day()
    stats = met.finalize()
    stats["seconds"] = time.time() - t0
    stats["policy"] = policy
    stats["cohorts"] = {"sessions_per_day": bots.sessions_per_day, "breadth": bots.breadth}
    log(f"[swarm] {policy:<18} iwe/bot={stats['iwe_mean']:.3f} imp={stats['impressions']:,} "
        f"sessions={len(np.concatenate(met.sess_len)) if met.sess_len else 0:,} "
        f"in {stats['seconds']:.1f}s")
    return stats


def _run_wave(world, fs, resp, met, policy, scorer, sess_bots, d, w, cap, cold_max,
              collect_metrics, collector=None) -> None:
    cfg = world.cfg
    bots, cat = world.bots, world.catalog
    n_sess = len(sess_bots)
    sess_hour = bots.hour[sess_bots, d, w].astype(np.int64)
    ts = d * DAY_MS + sess_hour * HOUR_MS
    pool = build_pool(world, fs, sess_bots, ts, world.rng(7, d, w), cfg["policy"]["retrieval"])
    comp = None
    if policy in ("taste_composer", "oracle"):
        if policy == "taste_composer":
            base, prior = taste_with_prior(world, fs, sess_bots, pool, ts, scorer)
        else:
            base = rank_scores("oracle", world, fs, sess_bots, pool, ts, world.rng(8, d, w), scorer)
            prior = None
        comp = ComposerState(zscore_rows(base), cat.cat[pool], fs.item_n[pool], prior,
                             fs.item_n[pool] <= cold_max, cat.n_cats, cfg["policy"]["composer"])
        order = None
    else:
        order = order_from_scores(rank_scores(policy, world, fs, sess_bots, pool, ts,
                                              world.rng(8, d, w), scorer), cap)
    uw = world.rng(6, d, w).random((bots.n_bots, cap, 7), dtype=np.float32)
    sat_w = int(cfg["response"]["satiation_window"])
    sess_active = np.ones(n_sess, bool)
    sess_len = np.zeros(n_sess, np.int32)
    sess_pos = np.zeros(n_sess, np.int32)
    sess_hi = np.zeros(n_sess, np.int32)
    budget = bots.patience[sess_bots, d, w].astype(np.float64)
    hist = np.full((n_sess, sat_w), -1, np.int64)
    hist_n = np.zeros(n_sess, np.int64)
    for i in range(cap):
        act = np.flatnonzero(sess_active)
        if len(act) == 0:
            break
        pick = comp.select(act, world.rng(9, d, w, i)) if comp is not None else order[act, i]
        items = pool[act, pick]
        b = sess_bots[act]
        cats = cat.cat[items].astype(np.int64)
        vcount = np.minimum(hist_n[act], sat_w)
        sat = (hist[act] == cats[:, None]).sum(1) / np.maximum(vcount, 1)
        prior_n, _ = fs.seen_counts(b, items)
        match, q = world.appeal(b, items)
        out = resp.respond(match, q, sess_hour[act],
                           sat, prior_n, np.full(len(act), i / cap),
                           bots.like_mult[b], bots.follow_mult[b], bots.forward_mult[b],
                           uw[b, i, :])
        if collector is not None:
            x, ccat = fs.pair_features(b, items, ts[act])
            collector(x, ccat, out, b)
        if collect_metrics:
            cold = fs.item_n[items] <= cold_max
            met.step(b, items, cats, out, prior_n, cold, sess_hour[act])
        else:
            met.step(b, items, cats, out, prior_n, np.zeros(len(act), bool), sess_hour[act])
        fs.update(b, items, ts[act], out["tier"], out)
        sess_len[act] += 1
        sess_pos[act] += out["pos"].astype(np.int32)
        sess_hi[act] += (out["like"] | out["follow"] | out["forward"]).astype(np.int32)
        idx = hist_n[act] % sat_w
        hist[act, idx] = cats
        hist_n[act] += 1
        if comp is not None:
            comp.observe(act, pick, out["pos"].astype(np.float32))
        budget[act], leave = leave_step(budget[act], out["tier"], prior_n > 0,
                                        cfg["leave"])
        sess_active[act[leave | (i == cap - 1)]] = False
    met.add_sessions(sess_len, sess_pos, sess_hi)


def simulate(policy: str, cfg: dict, seed: int, scorer: TasteScorer | None = None,
             n_bots: int | None = None, n_days: int | None = None,
             collect_metrics: bool = True, collector=None, log=print) -> dict:
    world = build_world(cfg, seed, n_bots=n_bots, n_days=n_days)
    world.cfg["policy"] = cfg["policy"]
    return run_policy(world, policy, scorer, collect_metrics=collect_metrics,
                      collector=collector, log=log)
