"""Intuition: Fit the swarm's free constants so the simulated marginals reproduce the real ones —
one scalar at a time against its own target, on a small fast swarm, with every target left untouched.

Writes artifacts/calibration.json (constants only) and notes/calibration.md (sim vs real table).
"""
from __future__ import annotations

import copy
import json
import time

import yaml

from feed.swarm import simulate
from feed.world import ART, ROOT, load_config

CALIBRATION_POLICY = "popularity"
LOG = lambda *a, **k: None


def load_policy() -> dict:
    with open(ROOT / "config" / "policy.yaml") as f:
        return yaml.safe_load(f)


def evaluate(base: dict, params: dict, seed: int, n_bots: int, n_days: int) -> dict:
    cfg = copy.deepcopy(base)
    for block, kv in params.items():
        cfg[block].update(kv)
    return simulate(CALIBRATION_POLICY, cfg, seed, n_bots=n_bots, n_days=n_days, log=LOG)


def _metric(stats: dict, name: str) -> float:
    ps = stats["print_summary"]
    if name == "hour_peak":
        return max(ps["hour_click_rate"].values())
    if name == "hour_trough":
        return min(ps["hour_click_rate"].values())
    if name == "engagement_by_len_high":
        return float(ps["engagement_by_len"]["31-100"])
    if name == "engagement_by_len_low":
        return float(ps["engagement_by_len"]["1"])
    return float(ps[name])


def bisect(base: dict, params: dict, block: str, key: str, name: str, lo: float, hi: float,
           target: float, increasing: bool, seed: int, n_bots: int, n_days: int,
           iters: int = 14, tol: float = 0.002) -> float:
    best = (lo + hi) / 2.0
    for _ in range(iters):
        mid = (lo + hi) / 2.0
        params[block][key] = mid
        val = _metric(evaluate(base, params, seed, n_bots, n_days), name)
        best = mid
        if abs(val - target) <= tol:
            break
        if (val < target) == increasing:
            lo = mid
        else:
            hi = mid
    params[block][key] = best
    return best


def fit(seed: int = 42, n_bots: int = 2000, n_days: int = 3, passes: int = 3) -> dict:
    base = load_config(use_calibration=False)
    base["policy"] = load_policy()
    targets = base["calibration"]["targets"]
    params = {"response": {}, "leave": {}}
    t0 = time.time()
    for p in range(passes):
        bisect(base, params, "leave", "patience_median", "median_session_items", 3.0, 20.0,
               float(targets["median_session_items"]), True, seed, n_bots, n_days, tol=5e-4)
        bisect(base, params, "leave", "reward_satisfying", "engagement_by_len_high", 0.0, 1.2,
               targets["engagement_by_len_high"], True, seed, n_bots, n_days, tol=1e-3)
        bisect(base, params, "leave", "penalty_skip", "engagement_by_len_low", 0.2, 2.0,
               targets["engagement_by_len_low"], False, seed, n_bots, n_days, tol=1e-3)
        bisect(base, params, "response", "p_like_base", "like_given_click", 0.005, 0.15,
               targets["like_given_click"], True, seed, n_bots, n_days, tol=5e-4)
        bisect(base, params, "response", "p_follow_base", "follow_given_click", 0.0005, 0.03,
               targets["follow_given_click"], True, seed, n_bots, n_days, tol=3e-4)
        bisect(base, params, "response", "p_forward_base", "forward_given_click", 0.0005, 0.03,
               targets["forward_given_click"], True, seed, n_bots, n_days, tol=3e-4)
        bisect(base, params, "response", "c0", "feed_click_rate", -2.0, 2.0,
               targets["feed_click_rate"], True, seed, n_bots, n_days, tol=5e-4)
        _fit_hour(base, params, targets, seed, n_bots, n_days)
        bisect(base, params, "response", "repeat_penalty", "repeat_click_mult_1", 0.0, 6.0,
               targets["repeat_click_mult_1"], False, seed, n_bots, n_days, tol=1e-3)
        print(f"[calibrate] pass {p + 1}/{passes} done in {time.time() - t0:.0f}s  {params}")
    stats = evaluate(base, params, seed, n_bots, n_days)
    return {"params": params, "stats": stats, "base": base, "targets": targets,
            "seconds": time.time() - t0, "seed": seed, "n_bots": n_bots, "n_days": n_days}


def _fit_hour(base: dict, params: dict, targets: dict, seed: int, n_bots: int, n_days: int) -> None:
    want = (targets["hour_peak"], targets["hour_trough"])
    best_amp, best_err = 1.0, float("inf")
    for amp in [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0]:
        params["response"]["hour_amp"] = amp
        st = evaluate(base, params, seed, n_bots, n_days)
        err = abs(_metric(st, "hour_peak") - want[0]) + abs(_metric(st, "hour_trough") - want[1])
        if err < best_err:
            best_err, best_amp = err, amp
    params["response"]["hour_amp"] = best_amp


def report(out: dict) -> dict:
    st = out["stats"]
    ps, gl, dg = st["print_summary"], st["guardrails"], st["diagnostics"]
    tg = out["targets"]

    def ok(value, target):
        return abs(value - target) / max(abs(target), 1e-9) <= tg_tol

    tg_tol = 0.15
    rows = [
        ("feed_click_rate", ps["feed_click_rate"], tg["feed_click_rate"]),
        ("like|click", ps["like_given_click"], tg["like_given_click"]),
        ("follow|click", ps["follow_given_click"], tg["follow_given_click"]),
        ("forward|click", ps["forward_given_click"], tg["forward_given_click"]),
        ("hour peak (click)", max(ps["hour_click_rate"].values()), tg["hour_peak"]),
        ("hour trough (click)", min(ps["hour_click_rate"].values()), tg["hour_trough"]),
        ("repeat x1", ps["repeat_click_mult_1"], tg["repeat_click_mult_1"]),
        ("repeat x2", ps["repeat_click_mult_2"], tg["repeat_click_mult_2"]),
        ("repeat x3", ps["repeat_click_mult_3"], tg["repeat_click_mult_3"]),
        ("median session", ps["median_session_items"], tg["median_session_items"]),
        ("eng 1-item", ps["engagement_by_len"]["1"], tg["engagement_by_len_low"]),
        ("eng 31-100", ps["engagement_by_len"]["31-100"], tg["engagement_by_len_high"]),
    ]
    checks = {name: {"sim": float(v), "target": float(t), "pass": bool(ok(v, t)),
                     "rel_err": float(abs(v - t) / max(abs(t), 1e-9))} for name, v, t in rows}
    return {"checks": checks,
            "pass_rate": float(sum(c["pass"] for c in checks.values()) / len(checks)),
            "iwe_mean": st["iwe_mean"], "guardrails": gl, "diagnostics": dg}


def main() -> None:
    t0 = time.time()
    out = fit()
    params = out["params"]
    cal = {"response": params["response"], "leave": params["leave"],
           "policy_used": CALIBRATION_POLICY, "seed": out["seed"],
           "n_bots": out["n_bots"], "n_days": out["n_days"], "seconds": out["seconds"]}
    with open(ART / "calibration.json", "w") as f:
        json.dump(cal, f, indent=1)
    rep = report(out)
    with open(ART / "calibration_report.json", "w") as f:
        json.dump(rep, f, indent=1)
    print(f"[calibrate] {rep['pass_rate'] * 100:.0f}% targets within 15% in {time.time() - t0:.0f}s")
    for name, c in rep["checks"].items():
        print(f"[calibrate]   {'PASS' if c['pass'] else 'FAIL'} {name:<20} "
              f"sim={c['sim']:.4f} target={c['target']:.4f}")


if __name__ == "__main__":
    main()
