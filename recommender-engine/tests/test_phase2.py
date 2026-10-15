"""Intuition: phase-2 guardrails — the world must rebuild deterministically, the response must stay
in distribution, policies must share random numbers, and the live feature path must match offline.
"""
import numpy as np
import pytest

from feed.metrics import Metrics
from feed.parity import parity_check
from feed.policies import rank_scores
from feed.response import ResponseModel, leave_step, return_prob
from feed.swarm import simulate
from feed.world import build_world, load_config
from run_swarm import load_policy


@pytest.fixture(scope="module")
def cfg():
    c = load_config()
    c["policy"] = load_policy()
    return c


def test_world_rebuilds_deterministically(cfg):
    a = build_world(cfg, 42, n_bots=300, n_days=2)
    b = build_world(cfg, 42, n_bots=300, n_days=2)
    assert np.array_equal(a.catalog.q, b.catalog.q)
    assert np.array_equal(a.bots.taste, b.bots.taste)
    assert np.array_equal(a.bots.counts, b.bots.counts)
    assert np.array_equal(a.bots.patience, b.bots.patience)
    assert np.all(np.isfinite(a.bots.taste))
    assert np.allclose(a.bots.taste.sum(1), 1.0, atol=1e-5)


def test_bot_schedule_is_policy_independent(cfg):
    r1 = simulate("random", cfg, 42, None, n_bots=400, n_days=2, log=lambda *a: None)
    r2 = simulate("popularity", cfg, 42, None, n_bots=400, n_days=2, log=lambda *a: None)
    assert r1["cohorts"]["sessions_per_day"].shape == r2["cohorts"]["sessions_per_day"].shape
    assert np.array_equal(r1["cohorts"]["breadth"], r2["cohorts"]["breadth"])


def test_response_bounds_and_tiers(cfg):
    rm = ResponseModel(cfg)
    n = 5000
    rng = np.random.default_rng(0)
    match = rng.random(n).astype(np.float32)
    q = rng.random(n).astype(np.float32)
    hour = rng.integers(0, 24, n)
    out = rm.respond(match, q, hour, rng.random(n).astype(np.float32),
                     rng.integers(0, 4, n).astype(np.int32), rng.random(n).astype(np.float32),
                     np.ones(n), np.ones(n), np.ones(n), rng.random((n, 7)))
    p = rm.p_click(match, q, hour, np.zeros(n), np.zeros(n, np.int32), np.zeros(n))
    assert np.all((p > 0) & (p < 1))
    assert set(np.unique(out["tier"])) <= {0, 1, 3, 4}
    assert np.all(out["watch"][~out["click"]] == 0.0)
    assert np.all((out["watch"] >= 0) & (out["watch"] <= 1.0))


def test_leave_and_return_bounds(cfg):
    budget = np.full(100, 2.0)
    _, left = leave_step(budget.copy(), np.zeros(100, np.int8), np.zeros(100, bool), cfg["leave"])
    assert left.dtype == bool
    prob = return_prob(np.full(50, 0.5), np.zeros(50), np.zeros(50), cfg["return_model"])
    assert np.all((prob > 0) & (prob < 1))


def test_metrics_iwe_math(cfg):
    m = Metrics(2, 10, 3, cfg["iwe"], 10)
    out = {"click": np.array([True, False]), "long_view": np.array([True, False]),
           "like": np.array([True, False]), "follow": np.array([False, False]),
           "forward": np.array([False, True]), "hate": np.array([False, False]),
           "pos": np.array([1.0, 1.0]), "score": np.array([6.0, 4.0]), "tier": np.array([3, 4])}
    m.step(np.array([0, 1]), np.array([0, 1]), np.array([0, 1]), out, np.zeros(2, np.int32),
           np.zeros(2, bool), np.array([12, 12]))
    st = m.finalize()
    assert np.isclose(st["iwe_per_bot"][0], 1 + 2 + 3)
    assert np.isclose(st["iwe_per_bot"][1], 4)


def test_rank_scores_isolate_retrieval(cfg):
    w = build_world(cfg, 42, n_bots=200, n_days=2)
    from feed.featurestate import FeatureState
    fs = FeatureState(w.catalog, w.bots.n_bots)
    bots = np.arange(50)
    ts = np.zeros(50)
    pool = np.tile(np.arange(w.catalog.n_items // 200), (50, 1))
    r = rank_scores("random", w, fs, bots, pool, ts, w.rng(8, 0, 0))
    p = rank_scores("popularity", w, fs, bots, pool, ts, w.rng(8, 0, 0))
    o = rank_scores("oracle", w, fs, bots, pool, ts, w.rng(8, 0, 0))
    assert r.shape == p.shape == o.shape == pool.shape
    assert np.all(np.isfinite(o))


def test_appeal_is_user_conditional_and_deterministic(cfg):
    a = build_world(cfg, 42, n_bots=300, n_days=2)
    b = build_world(cfg, 42, n_bots=300, n_days=2)
    assert np.allclose(a.catalog.idio, b.catalog.idio)
    assert np.allclose(a.bots.pref_cat, b.bots.pref_cat)
    assert np.allclose(a.bots.pref_idio, b.bots.pref_idio)
    items = np.arange(200)
    m0, q0 = a.appeal(np.zeros(200, int), items)
    m1, q1 = a.appeal(np.ones(200, int), items)
    assert np.all(np.isfinite(m0)) and np.all(np.isfinite(q0))
    assert not np.allclose(m0, m1)  # same items, different bots -> different appeal
    assert np.array_equal(q0, q1)   # global quality is user-independent
    corr = np.corrcoef(a.bots.pref_cat[0], a.bots.taste[0])[0, 1]
    assert corr > 0.9               # category preference tracks the real decay share


def test_composer_prior_seen_penalty(cfg):
    from feed.composer import ComposerState
    z = np.array([[1.0, 0.9, 0.8, 0.7]], np.float32)
    inv_i = np.ones((1, 4), np.float64)
    zero = np.zeros((1, 4), np.float32)
    knobs = {"gamma": 0.0, "alpha": 0.0, "eta": 0.0, "beta": 0.0, "epsilon": 0.0,
             "decay": 0.8, "window_m": 5.0}
    rng = np.random.default_rng(0)
    cold = np.zeros((1, 4), bool)
    c0 = ComposerState(z, np.zeros((1, 4), int), inv_i, zero, cold, 3, knobs)
    assert c0.select(np.array([0]), rng)[0] == 0
    prior = np.array([[3.0, 0.0, 0.0, 0.0]], np.float32)
    c1 = ComposerState(z, np.zeros((1, 4), int), inv_i, prior, cold, 3,
                       {**knobs, "gamma": 2.0})
    assert c1.select(np.array([0]), rng)[0] == 1


def test_parity_gate(cfg):
    res = parity_check(cfg, log=lambda *a: None)
    assert res["status"] == "ok" and res["pass"], res.get("checks")
