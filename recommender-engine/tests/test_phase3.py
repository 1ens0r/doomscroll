"""Intuition: Phase-3 guardrails — the composer generation accept/reject rule must be directional
and strict (a component is kept only on a real IWE gain with no guardrail degradation).
"""
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
    assert len(GENERATIONS) == 5
    assert [g[1] for g in GENERATIONS] == ["seen-penalty", "diversity", "session adaptation",
                                           "cold bonus", "epsilon exploration"]
    assert set(NEUTRAL) >= {"gamma", "alpha", "eta", "beta", "epsilon", "decay", "window_m"}
