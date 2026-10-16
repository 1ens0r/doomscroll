"""Intuition: Phase-3 guardrails — the composer generation accept/reject rule must be directional
and strict (a component is kept only on a real IWE gain with no guardrail degradation).
"""
import numpy as np
import pytest

from run_generations import GENERATIONS, NEUTRAL, guardrail_ok


def test_guardrail_direction_and_threshold():
    base = {"cat_entropy_bits": 4.0, "catalog_coverage": 0.05,
            "cold_impression_share": 0.01, "repeat_exposure_rate": 0.01}
    better = {"cat_entropy_bits": 4.2, "catalog_coverage": 0.05,
              "cold_impression_share": 0.01, "repeat_exposure_rate": 0.009}
    assert guardrail_ok(base, better, 0.10)["pass"]
    worse_repeat = {**base, "repeat_exposure_rate": 0.02}  # +100% is bad
    assert not guardrail_ok(base, worse_repeat, 0.10)["pass"]
    worse_cov = {**base, "catalog_coverage": 0.04}          # -20% is bad
    assert not guardrail_ok(base, worse_cov, 0.10)["pass"]
    tiny = {**base, "catalog_coverage": 0.047}              # -6% is within tolerance
    assert guardrail_ok(base, tiny, 0.10)["pass"]


def test_generation_plan_shape():
    assert len(GENERATIONS) == 8
    assert [g[1] for g in GENERATIONS] == ["seen-penalty", "diversity", "session adaptation",
                                           "cold bonus", "epsilon exploration",
                                           "recency-weighted diversity", "length-adaptive diversity",
                                           "recency-boosted diversity (joint retune)"]
    assert set(NEUTRAL) >= {"gamma", "alpha", "eta", "beta", "epsilon", "decay", "window_m",
                            "alpha_recency", "alpha_growth"}


def test_recency_weighting_changes_which_category_is_avoided():
    from feed.composer import ComposerState
    taste_z = np.zeros((1, 6), dtype=np.float32)
    pool_cat = np.array([[0, 1, 2, 3, 4, 5]], dtype=np.int64)
    pool_i_n = np.zeros((1, 6), dtype=np.int64)
    base = {"gamma": 0.0, "alpha": 0.6, "eta": 0.0, "beta": 0.0, "epsilon": 0.0,
            "decay": 0.8, "window_m": 3}
    act = np.array([0])

    def same_cat(recency):
        cs = ComposerState(taste_z, pool_cat, pool_i_n, None, np.zeros((1, 6), bool), 6,
                           {**base, "alpha_recency": recency, "alpha_growth": 0.0})
        cs.observe(act, np.array([0]), np.array([1.0], dtype=np.float32))  # cat 0, older
        cs.observe(act, np.array([1]), np.array([1.0], dtype=np.float32))  # cat 1, recent
        return cs._same_cat(act)[0]

    count = same_cat(1.0)
    recency = same_cat(0.2)
    # Count penalty treats both seen categories equally; recency weights the recent one harder.
    assert count[0] == 1.0 and count[1] == 1.0 and count[2] == 0.0
    assert recency[0] == pytest.approx(0.2) and recency[1] == pytest.approx(1.0)
    assert recency[2] == 0.0


def test_length_adaptive_diversity_growth():
    from feed.composer import ComposerState
    taste_z = np.zeros((1, 6), dtype=np.float32)
    pool_cat = np.array([[0, 1, 2, 3, 4, 5]], dtype=np.int64)
    pool_i_n = np.zeros((1, 6), dtype=np.int64)
    base = {"gamma": 0.0, "alpha": 0.3, "eta": 0.0, "beta": 0.0, "epsilon": 0.0,
            "decay": 0.8, "window_m": 4}
    act = np.array([0])
    cs = ComposerState(taste_z, pool_cat, pool_i_n, None, np.zeros((1, 6), bool), 6,
                       {**base, "alpha_recency": 1.0, "alpha_growth": 1.0})
    for _ in range(3):
        cs.observe(act, np.array([0]), np.array([0.0], dtype=np.float32))
    # With growth, the re-seen category is penalised; an unseen category still wins.
    assert cs.select(act, np.random.default_rng(0))[0] != 0
