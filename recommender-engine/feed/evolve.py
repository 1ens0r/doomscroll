"""Intuition: Phase-4 evolution — a (1+lambda) evolution strategy searches the policy knobs
(composer coefficients, score weights, retrieval mix) for swarm IWE, penalised by guardrail
violations so engagement is never bought by collapsing coverage or diversity. Search on TRAIN,
select on VAL, measure the winner on TEST exactly once.
"""
from __future__ import annotations

import copy

import numpy as np

# knob name -> (policy block, key inside that block)
KNOB_BLOCKS: dict[str, tuple[str, str]] = {
    "gamma": ("composer", "gamma"),
    "alpha": ("composer", "alpha"),
    "eta": ("composer", "eta"),
    "beta": ("composer", "beta"),
    "epsilon": ("composer", "epsilon"),
    "d": ("composer", "decay"),
    "m": ("composer", "window_m"),
    "w_click": ("score_weights", "click"),
    "w_watch": ("score_weights", "watch"),
    "w_like": ("score_weights", "like"),
    "w_forward": ("score_weights", "forward"),
    "w_follow": ("score_weights", "follow"),
    "n_popular": ("retrieval", "n_popular"),
    "n_affinity": ("retrieval", "n_affinity"),
    "n_new": ("retrieval", "n_new"),
    "n_random": ("retrieval", "n_random"),
}
INT_KEYS = {"m", "n_popular", "n_affinity", "n_new", "n_random"}
RETRIEVAL_KEYS = ("n_popular", "n_affinity", "n_new", "n_random")
# Guardrail direction: +1 = higher is better, -1 = lower is better.
GUARDRAIL_DIR = {"cat_entropy_bits": 1, "catalog_coverage": 1,
                 "cold_impression_share": 1, "repeat_exposure_rate": -1}


def start_vector(spec: dict) -> np.ndarray:
    return np.array([float(spec[k]["value"]) for k in spec], dtype=np.float64)


def knobs_to_vector(knobs: dict, spec: dict) -> np.ndarray:
    return np.array([float(knobs[k]) for k in spec], dtype=np.float64)


def bounds(spec: dict) -> tuple[np.ndarray, np.ndarray]:
    lo = np.array([float(spec[k]["min"]) for k in spec], dtype=np.float64)
    hi = np.array([float(spec[k]["max"]) for k in spec], dtype=np.float64)
    return lo, hi


def vector_to_knobs(x: np.ndarray, spec: dict) -> dict:
    """Clip to bounds and round integer knobs; this is the candidate's canonical form."""
    out = {}
    for i, k in enumerate(spec):
        v = float(np.clip(x[i], float(spec[k]["min"]), float(spec[k]["max"])))
        out[k] = int(round(v)) if k in INT_KEYS else v
    return out


def mutate(x: np.ndarray, spec: dict, sigma: float, rng: np.random.Generator) -> np.ndarray:
    """Gaussian mutation scaled by `sigma` x each knob's search range, clipped to its bounds."""
    lo, hi = bounds(spec)
    return np.clip(x + sigma * (hi - lo) * rng.standard_normal(len(x)), lo, hi)


def knobs_to_policy(policy: dict, knobs: dict) -> dict:
    """Deep-copy the policy and install a candidate's composer / score weights / retrieval mix."""
    p = copy.deepcopy(policy)
    for name, (block, key) in KNOB_BLOCKS.items():
        p[block][key] = knobs[name]
    rc = p["retrieval"]
    rc["pool_size"] = int(sum(int(round(rc[k])) for k in RETRIEVAL_KEYS))
    return p


def guardrail_violations(ref: dict, cand: dict, tol: float) -> int:
    """Count guardrails that moved >tol in their bad direction relative to the reference."""
    n = 0
    for k, direction in GUARDRAIL_DIR.items():
        b, c = float(ref[k]), float(cand[k])
        rel = (c - b) / max(abs(b), 1e-9)
        if (direction == 1 and rel < -tol) or (direction == -1 and rel > tol):
            n += 1
    return n


def fitness(iwe: float, ref_guard: dict, cand_guard: dict, tol: float,
            penalty_frac: float) -> tuple[float, int]:
    """IWE discounted by `penalty_frac` of its own value per violated guardrail."""
    nv = guardrail_violations(ref_guard, cand_guard, tol)
    return float(iwe) * (1.0 - float(penalty_frac) * nv), nv


def build_task(cfg: dict, policy: dict, knobs: dict, seed: int, n_bots: int, n_days: int,
               threads: int) -> dict:
    """Picklable unit of work for a process pool."""
    c = copy.deepcopy(cfg)
    c["policy"] = knobs_to_policy(policy, knobs)
    return {"cfg": c, "policy": c["policy"], "knobs": dict(knobs), "seed": int(seed),
            "n_bots": int(n_bots), "n_days": int(n_days), "threads": int(threads)}


def evaluate_task(task: dict) -> dict:
    """Worker: simulate one candidate on a swarm and return IWE + guardrails.

    Safe in a `multiprocessing` pool; forces single-threaded torch so 8 workers do not oversubscribe.
    """
    import random

    import torch

    seed = int(task["seed"])
    torch.set_num_threads(int(task["threads"]))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    from feed.policies import TasteScorer
    from feed.swarm import simulate

    scorer = TasteScorer(task["policy"], task["threads"])
    stats = simulate("taste_composer", task["cfg"], seed, scorer, n_bots=int(task["n_bots"]),
                     n_days=int(task["n_days"]), log=lambda *a: None)
    return {"knobs": task["knobs"], "iwe": float(stats["iwe_mean"]),
            "guardrails": stats["guardrails"], "iwe_per_bot": np.asarray(stats["iwe_per_bot"]),
            "impressions": int(stats["impressions"]), "seconds": float(stats["seconds"])}
