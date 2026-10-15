"""Intuition: Phase-2 entry point: run every policy on the same bots/world and check the IWE ladder
random < popularity < personal_affinity < taste < oracle with paired per-bot bootstrap CIs.
"""
from __future__ import annotations

import argparse
import json
import random
import time

import numpy as np
import torch
import yaml

from feed.eval import paired_bootstrap
from feed.parity import parity_check
from feed.policies import TasteScorer
from feed.swarm import POLICIES, simulate
from feed.world import ROOT, load_config

MODELS = ROOT / "models"
POLICY_PATH = ROOT / "config" / "policy.yaml"
ART = ROOT / "artifacts"
LADDER = ["random", "popularity", "personal_affinity", "taste", "oracle"]
NEEDS_SCORER = {"taste", "taste_composer"}


def load_policy() -> dict:
    with open(POLICY_PATH) as f:
        policy = yaml.safe_load(f)
    policy["_models_dir"] = MODELS
    # Prefer the ranker learned on simulator logs (Learn stage) when it exists.
    policy["_taste_ckpt"] = "taste_sim.pt" if (MODELS / "taste_sim.pt").exists() else "taste.pt"
    return policy


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _clean(obj):
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    return obj


def _terciles(x: np.ndarray) -> np.ndarray:
    return np.digitize(x, np.quantile(x, [1 / 3, 2 / 3]))


def cohorts(stats_by_policy: dict[str, dict], policies: list[str]) -> dict:
    spd = stats_by_policy[policies[0]]["cohorts"]["sessions_per_day"]
    brd = stats_by_policy[policies[0]]["cohorts"]["breadth"]
    act, bre = _terciles(np.asarray(spd, float)), _terciles(np.asarray(brd, float))
    out = {}
    for pi in policies:
        iwe = stats_by_policy[pi]["iwe_per_bot"]
        cell = {}
        for a in range(3):
            for b in range(3):
                m = (act == a) & (bre == b)
                cell[f"A{a}_B{b}"] = float(iwe[m].mean()) if m.any() else float("nan")
        cell["heavy_users"] = float(iwe[act == 2].mean())
        cell["light_users"] = float(iwe[act == 0].mean())
        out[pi] = cell
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--policy", default="all", choices=["all"] + POLICIES)
    ap.add_argument("--bots", type=int, default=None)
    ap.add_argument("--days", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--n-boot", type=int, default=None)
    ap.add_argument("--no-parity", action="store_true")
    args = ap.parse_args()

    cfg = load_config()
    cfg["policy"] = load_policy()
    seed = int(args.seed if args.seed is not None else cfg["seed"])
    n_bots = args.bots if args.bots else (2000 if args.fast else int(cfg["swarm"]["bots"]))
    n_days = args.days if args.days else (3 if args.fast else int(cfg["swarm"]["days"]))
    n_boot = args.n_boot or (500 if args.fast else int(cfg["metrics"]["bootstrap_resamples"]))
    seed_all(seed)

    policies = LADDER + ["taste_composer"] if args.policy == "all" else [args.policy]
    scorer = TasteScorer(cfg["policy"], cfg["policy"]["python_threads"]) if NEEDS_SCORER.intersection(
        policies) else None

    t0 = time.time()
    stats, times = {}, {}
    for policy in policies:
        t1 = time.time()
        stats[policy] = simulate(policy, cfg, seed, scorer, n_bots=n_bots, n_days=n_days)
        times[policy] = time.time() - t1
    iwe = {p: np.asarray(stats[p]["iwe_per_bot"], float) for p in policies}

    pairs, ladder_ok = {}, True
    for a, b in zip(LADDER[:-1], LADDER[1:]):
        ci = paired_bootstrap(iwe[b], iwe[a], n_boot, seed)
        ci["pass"] = bool(ci["lo"] > 0)
        ladder_ok &= ci["pass"]
        pairs[f"{b}_gt_{a}"] = ci
        print(f"[swarm] {b:>16} > {a:<16} d={ci['mean_diff']:+.4f} "
              f"[{ci['lo']:+.4f},{ci['hi']:+.4f}] {'PASS' if ci['pass'] else 'FAIL'}")

    base, ceil = iwe["random"].mean(), iwe["oracle"].mean()
    gap = {p: (float(iwe[p].mean() - base) / (ceil - base)) for p in policies if ceil != base}

    result = {
        "status": "ok", "config": {"seed": seed, "bots": n_bots, "days": n_days,
                                   "calibrated": cfg["_calibrated"], "n_boot": n_boot},
        "iwe": {p: {"mean": float(iwe[p].mean()), "std": float(iwe[p].std())} for p in policies},
        "oracle_gap_closed": gap, "times_sec": times, "ladder": pairs,
        "acceptance": {"pass": bool(ladder_ok), "order": LADDER},
        "print_summary": {p: stats[p]["print_summary"] for p in policies},
        "diagnostics": {p: stats[p]["diagnostics"] for p in policies},
        "guardrails": {p: stats[p]["guardrails"] for p in policies},
        "cohorts": cohorts(stats, policies),
        "seconds_total": time.time() - t0,
    }
    if not args.no_parity:
        pcfg = {**cfg, "parity": cfg["parity"]}
        result["parity"] = parity_check(pcfg)
    ART.mkdir(exist_ok=True)
    with open(ART / "sim_results.json", "w") as f:
        json.dump(_clean(result), f, indent=2)
    print(f"[swarm] acceptance {'PASS' if ladder_ok else 'FAIL'} | "
          f"oracle-gap-closed @taste={gap.get('taste', float('nan')):.2f} | wrote sim_results.json")
    return 0 if ladder_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
