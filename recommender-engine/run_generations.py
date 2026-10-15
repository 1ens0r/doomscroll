"""Intuition: Phase-3 — grow the Compose stage one component at a time. Start from a neutral
composer (no re-ordering beyond the ranker), then try each component on top of the accepted set;
keep it only if IWE rises with a 95% paired CI that excludes zero and no guardrail degrades >10%.
Rejections are logged with their numbers, not hidden.
"""
from __future__ import annotations

import argparse
import copy
import json
import time

import numpy as np

from feed.eval import paired_bootstrap
from feed.policies import TasteScorer
from feed.swarm import simulate
from feed.world import ART, load_config
from run_swarm import _clean, load_policy, seed_all

# (generation, component, knob, on-value, hypothesis)
GENERATIONS = [
    (1, "seen-penalty", "gamma", 2.0,
     "down-rank items already shown in the session to clear repeat fatigue"),
    (2, "diversity", "alpha", 0.4,
     "penalise category runs so a session does not collapse into one topic"),
    (3, "session adaptation", "eta", 0.5,
     "boost categories the current session is actually rewarding"),
    (4, "cold bonus", "beta", 0.3,
     "up-rank under-exposed items via beta/sqrt(1+i_n)"),
    (5, "epsilon exploration", "epsilon", 0.05,
     "reserve an occasional cold-item slot instead of always exploiting"),
]
NEUTRAL = {"gamma": 0.0, "alpha": 0.0, "eta": 0.0, "beta": 0.0, "epsilon": 0.0,
           "decay": 0.8, "window_m": 5.0}
# Guardrail direction: +1 = higher is better, -1 = lower is better.
GUARDRAIL_DIR = {"cat_entropy_bits": 1, "catalog_coverage": 1,
                 "cold_impression_share": 1, "repeat_exposure_rate": -1}


def guardrail_ok(base: dict, cand: dict, tol: float) -> dict:
    """A guardrail degrades if it moves >tol in its bad direction."""
    out, ok = {}, True
    for k, direction in GUARDRAIL_DIR.items():
        b, c = float(base[k]), float(cand[k])
        scale = max(abs(b), 1e-9)
        rel = (c - b) / scale
        degraded = (direction == 1 and rel < -tol) or (direction == -1 and rel > tol)
        out[k] = {"base": b, "cand": c, "rel": float(rel), "degraded": bool(degraded)}
        ok &= not degraded
    return {"checks": out, "pass": bool(ok)}


def run_composer(cfg: dict, knobs: dict, scorer, n_bots: int, n_days: int, seed: int) -> dict:
    c = copy.deepcopy(cfg)
    c["policy"]["composer"] = {**c["policy"]["composer"], **knobs}
    return simulate("taste_composer", c, seed, scorer, n_bots=n_bots, n_days=n_days,
                    log=lambda *a: None)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--bots", type=int, default=None)
    ap.add_argument("--days", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--n-boot", type=int, default=None)
    ap.add_argument("--tol", type=float, default=0.10, help="max guardrail degradation")
    args = ap.parse_args()

    cfg = load_config()
    cfg["policy"] = load_policy()
    seed = int(args.seed if args.seed is not None else cfg["seed"])
    n_bots = args.bots if args.bots else (2000 if args.fast else int(cfg["swarm"]["bots"]))
    n_days = args.days if args.days else (3 if args.fast else int(cfg["swarm"]["days"]))
    n_boot = args.n_boot or (500 if args.fast else int(cfg["metrics"]["bootstrap_resamples"]))
    seed_all(seed)
    scorer = TasteScorer(cfg["policy"], cfg["policy"]["python_threads"])

    t0 = time.time()
    accepted = dict(NEUTRAL)
    base_stats = run_composer(cfg, accepted, scorer, n_bots, n_days, seed)
    base_iwe = np.asarray(base_stats["iwe_per_bot"], float)
    print(f"[gen0] neutral composer iwe/bot={base_iwe.mean():.3f}")

    rows, new_iwe, new_stats = [], base_iwe, base_stats
    for gen, component, knob, value, hypothesis in GENERATIONS:
        knobs = {**accepted, knob: value}
        st = run_composer(cfg, knobs, scorer, n_bots, n_days, seed)
        iwe = np.asarray(st["iwe_per_bot"], float)
        ci = paired_bootstrap(iwe, new_iwe, n_boot, seed)
        gr = guardrail_ok(new_stats["guardrails"], st["guardrails"], args.tol)
        keep = bool(ci["lo"] > 0 and gr["pass"])
        verdict = "KEEP" if keep else ("REJECT (no IWE gain)" if ci["lo"] <= 0
                                       else "REJECT (guardrail)")
        if keep:
            accepted = knobs
            new_iwe, new_stats = iwe, st
        rows.append({
            "generation": gen, "component": component, "knob": knob, "value": value,
            "hypothesis": hypothesis, "iwe": float(iwe.mean()),
            "delta_vs_accepted": float(ci["mean_diff"]),
            "ci": [float(ci["lo"]), float(ci["hi"])], "guardrails": gr,
            "kept": keep, "verdict": verdict})
        print(f"[gen{gen}] {component:<20} {knob}={value:<5} iwe={iwe.mean():.3f} "
              f"d={ci['mean_diff']:+.3f} [{ci['lo']:+.3f},{ci['hi']:+.3f}] "
              f"gr={'ok' if gr['pass'] else 'BAD'} -> {verdict}")

    final_iwe = new_iwe.mean()
    base, oracle = base_iwe.mean(), None
    result = {
        "status": "ok", "seed": seed, "bots": n_bots, "days": n_days, "n_boot": n_boot,
        "guardrail_tol": args.tol, "composer_neutral": NEUTRAL,
        "composer_accepted": accepted, "iwe_neutral": float(base),
        "iwe_final": float(final_iwe),
        "gain_vs_neutral": float(final_iwe - base),
        "generations": rows, "seconds_total": time.time() - t0,
    }
    with open(ART / "composer_generations.json", "w") as f:
        json.dump(_clean(result), f, indent=1)
    print(f"[gen] accepted knobs {accepted}")
    print(f"[gen] iwe {base:.2f} -> {final_iwe:.2f} (+{final_iwe - base:.2f}) "
          f"in {time.time() - t0:.0f}s -> artifacts/composer_generations.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
