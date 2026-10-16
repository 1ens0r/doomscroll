"""Intuition: Phase-4 evolution mechanics — knob vectors must clip/round to their declared bounds,
policy installation must hit every composer/score/retrieval slot, and the guardrail penalty must
point the right way before any swarm run is trusted.
"""
import numpy as np

from feed import evolve
from feed.policies import effective_policy

SPEC = {
    "gamma": {"value": 0.0, "min": 0.0, "max": 5.0},
    "m": {"value": 5.0, "min": 1.0, "max": 10.0},
    "w_like": {"value": 3.0, "min": 0.5, "max": 6.0},
    "n_popular": {"value": 50.0, "min": 10.0, "max": 90.0},
}


def test_vector_roundtrip_and_clipping():
    x = np.array([2.5, 7.7, 3.0, 50.0])
    knobs = evolve.vector_to_knobs(x, SPEC)
    assert knobs["gamma"] == 2.5
    assert knobs["m"] == 8 and isinstance(knobs["m"], int)
    assert knobs["n_popular"] == 50
    out = evolve.vector_to_knobs(np.array([-9.0, 99.0, 99.0, -1.0]), SPEC)
    assert out["gamma"] == 0.0 and out["m"] == 10 and out["w_like"] == 6.0
    assert out["n_popular"] == 10
    assert np.allclose(evolve.knobs_to_vector(knobs, SPEC), [2.5, 8.0, 3.0, 50.0])


def test_mutation_stays_in_bounds():
    rng = np.random.default_rng(0)
    lo, hi = evolve.bounds(SPEC)
    x = evolve.start_vector(SPEC)
    for _ in range(200):
        x = evolve.mutate(x, SPEC, 0.15, rng)
        assert np.all(x >= lo) and np.all(x <= hi)


def test_install_hits_every_policy_block():
    policy = {
        "composer": {"gamma": 0.0, "alpha": 0.4, "eta": 0.5, "beta": 0.0, "epsilon": 0.0,
                     "decay": 0.8, "window_m": 5},
        "score_weights": {"click": 1.0, "watch": 2.0, "like": 3.0, "forward": 4.0, "follow": 5.0},
        "retrieval": {"pool_size": 150, "n_popular": 50, "n_affinity": 50, "n_new": 30,
                      "n_random": 20},
    }
    full = {n: 1.0 for n in evolve.KNOB_BLOCKS}
    full["gamma"], full["alpha"] = 1.5, 0.9
    full["n_popular"], full["n_affinity"], full["n_new"], full["n_random"] = 40, 40, 25, 15
    full["m"] = 7
    p = evolve.knobs_to_policy(policy, full)
    assert p["composer"]["gamma"] == 1.5 and p["composer"]["alpha"] == 0.9
    assert p["composer"]["window_m"] == 7
    assert p["retrieval"]["pool_size"] == 120
    assert p["retrieval"]["n_random"] == 15
    assert policy["composer"]["gamma"] == 0.0  # original untouched


def test_fitness_penalises_guardrail_degradation():
    ref = {"cat_entropy_bits": 3.4, "catalog_coverage": 0.78,
           "cold_impression_share": 0.048, "repeat_exposure_rate": 0.005}
    same = dict(ref)
    fit_ok, nv_ok = evolve.fitness(100.0, ref, same, 0.10, 0.05)
    assert nv_ok == 0 and fit_ok == 100.0
    bad = {**ref, "catalog_coverage": 0.5}  # -36% coverage
    fit_bad, nv_bad = evolve.fitness(100.0, ref, bad, 0.10, 0.05)
    assert nv_bad == 1 and fit_bad == 95.0
    worse_repeat = {**ref, "repeat_exposure_rate": 0.006}  # +20% repeat
    assert evolve.guardrail_violations(ref, worse_repeat, 0.10) == 1


def test_effective_policy_overlays_evolved_only():
    base = {
        "composer": {"gamma": 0.0, "alpha": 0.4, "eta": 0.5, "beta": 0.0, "epsilon": 0.0,
                     "decay": 0.8, "window_m": 5},
        "score_weights": {"click": 1.0, "watch": 2.0, "like": 3.0, "forward": 4.0, "follow": 5.0},
        "retrieval": {"pool_size": 150, "n_popular": 50, "n_affinity": 50, "n_new": 30,
                      "n_random": 20},
        "_evolved": {"composer": {"alpha": 0.7},
                     "retrieval": {"n_popular": 30, "n_random": 10}},
    }
    assert effective_policy(base, "taste_composer") is base
    ev = effective_policy(base, "evolved")
    assert ev["composer"]["alpha"] == 0.7 and ev["composer"]["gamma"] == 0.0
    assert ev["retrieval"]["pool_size"] == 120
    assert base["composer"]["alpha"] == 0.4
