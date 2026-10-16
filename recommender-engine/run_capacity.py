"""Intuition: Phase 7 capacity study. Train config-driven TasteNet variants on the identical data,
split, seed, and protocol; select on validation NDCG@10 with paired-bootstrap CIs; touch test once."""
from __future__ import annotations

import argparse
import json
import random
import sys
import time

import numpy as np
import torch
import yaml

from feed.data import ART, ROOT, load_config
from feed.eval import paired_bootstrap
from feed.experiment import load_bundles, run_one
from feed.model import list_variants, load_meta

POLICY_PATH = ROOT / "config" / "policy.yaml"
MODELS = ROOT / "models"
RESULTS_DIR = ROOT.parent / "notes" / "results"

DEFAULT_VARIANTS = ["baseline", "wide", "wide_deep", "cross"]


def load_policy() -> dict:
    with open(POLICY_PATH) as f:
        return yaml.safe_load(f)


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def ci_of(a: dict, b: dict, k: int, n_boot: int, seed: int) -> dict:
    """Paired user bootstrap of mean(a - b) on val/feed for metric k."""
    ua, va = a["scopes"]["val"]["feed"]["taste"]["per_user"][str(k)]
    ub, vb = b["scopes"]["val"]["feed"]["taste"]["per_user"][str(k)]
    if not np.array_equal(ua, ub):
        raise RuntimeError("paired user mismatch")
    return paired_bootstrap(va, vb, n_boot, seed)


def print_table(results: list[dict], val_ci: dict, primary: int) -> None:
    print(f"\n== Capacity (val, feed, ndcg@{primary}) ==")
    print(f"{'variant':<12}{'params':>9}{'best_ep':>8}{'val@10':>9}{'val@20':>9}"
          f"{'train_s':>9}{'rss_MB':>8}   diff vs baseline [95% CI]")
    for r in results:
        v10 = r["scopes"]["val"]["feed"]["taste"][f"ndcg@{primary}"]["mean"]
        v20 = r["scopes"]["val"]["feed"]["taste"]["ndcg@20"]["mean"]
        c = val_ci.get(r["name"])
        ctxt = (f"{c['mean_diff']:+.4f} [{c['lo']:+.4f}, {c['hi']:+.4f}]"
                if c else "  (reference)")
        print(f"{r['name']:<12}{r['n_params']:>9,}{r['best_epoch']:>8}{v10:>9.4f}{v20:>9.4f}"
              f"{r['train_seconds']:>9.1f}{r['peak_rss_mb']:>8.0f}   {ctxt}")


def main() -> int:
    ap = argparse.ArgumentParser(description="Phase 7: model capacity experiments")
    ap.add_argument("--fast", action="store_true", help="1 epoch, 200k rows/variant, 500 boot")
    ap.add_argument("--variants", default=",".join(DEFAULT_VARIANTS))
    ap.add_argument("--test", default="", help="comma names to retrain and evaluate on test once")
    ap.add_argument("--max-rows", type=int, default=None)
    ap.add_argument("--boot", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--no-lgbm", action="store_true")
    args = ap.parse_args()

    t_start = time.time()
    policy = load_policy()
    seed = policy["seed"] if args.seed is None else args.seed
    seed_all(seed)
    torch.set_num_threads(int(policy["python_threads"]))
    meta = load_meta()
    gains = load_config()["tier_gains"]
    primary = policy["eval"]["primary_k"]
    if args.boot is not None:
        policy["eval"]["bootstrap_resamples"] = args.boot

    max_rows = args.max_rows
    if args.fast:
        max_rows = max_rows or 200_000
        policy["train"]["max_epochs"] = 1
        policy["eval"]["bootstrap_resamples"] = min(policy["eval"]["bootstrap_resamples"], 500)
    names = [n for n in args.test.split(",") if n] or [n for n in args.variants.split(",") if n]

    print(f"[capacity] known variants: {list_variants(policy)}")
    bundles = load_bundles(policy, max_rows=max_rows, seed=seed)
    print(f"[data] rows train={len(bundles['train']['x']):,} val={len(bundles['val']['x']):,} "
          f"test={len(bundles['test']['x']):,}")

    results = []
    for name in names:
        variant = None if name in ("baseline", "base") else name
        r = run_one(bundles, policy, meta, gains, name, variant=variant,
                    evaluate_test=bool(args.test), seed=seed,
                    ckpt_dir=MODELS / "capacity" / ("fast" if args.fast else "full"))
        print(f"[{name}] best_epoch={r['best_epoch']} val@10={r['best_val_ndcg10']:.4f} "
              f"train={r['train_seconds']:.1f}s")
        results.append(r)

    n_boot = policy["eval"]["bootstrap_resamples"]
    base = next((r for r in results if r["name"] == "baseline"), None)
    val_ci = {}
    if base is not None:
        for r in results:
            if r["name"] != "baseline":
                val_ci[r["name"]] = ci_of(r, base, primary, n_boot, seed)
    print_table(results, val_ci, primary)

    best = max(results, key=lambda r: r["scopes"]["val"]["feed"]["taste"][f"ndcg@{primary}"]["mean"])
    within = []
    for r in results:
        if r["name"] == best["name"]:
            within.append(r)
            continue
        c = ci_of(r, best, primary, n_boot, seed)
        if c["lo"] <= 0 <= c["hi"]:
            within.append(r)
    promoted = min(within, key=lambda r: r["n_params"]) if within else best
    print(f"\n[promote] best val = {best['name']}; within-CI set = "
          f"{[r['name'] for r in within]}; smallest params -> {promoted['name']}")

    payload = {
        "phase": 7, "fast": args.fast, "seed": seed, "max_rows": max_rows,
        "primary_k": primary,
        "variants": {r["name"]: {"n_params": r["n_params"], "best_epoch": r["best_epoch"],
                                 "train_seconds": r["train_seconds"],
                                 "peak_rss_mb": r["peak_rss_mb"], "history": r["history"],
                                 "pos_weights": r["pos_weights"]}
                     for r in results},
        "val_feed": {r["name"]: {"ndcg@10": r["scopes"]["val"]["feed"]["taste"]["ndcg@10"],
                                 "ndcg@20": r["scopes"]["val"]["feed"]["taste"]["ndcg@20"]}
                     for r in results},
        "paired_vs_baseline": val_ci,
        "best": best["name"], "within_ci": [r["name"] for r in within],
        "promoted": promoted["name"],
    }
    if args.test:
        payload["test_feed"] = {r["name"]: {"ndcg@10": r["scopes"]["test"]["feed"]["taste"]["ndcg@10"],
                                            "ndcg@20": r["scopes"]["test"]["feed"]["taste"]["ndcg@20"]}
                                for r in results}
    ART.mkdir(exist_ok=True)
    out = ART / ("capacity_results_fast.json" if args.fast else "capacity_results.json")
    out.write_text(json.dumps(payload, indent=1))
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "capacity.json").write_text(json.dumps(
        {**{k: v for k, v in payload.items() if k != "variants"},
         "variants": {n: {kk: vv for kk, vv in v.items() if kk != "history"}
                      for n, v in payload["variants"].items()}},
        indent=1))
    print(f"[done] {out} + {RESULTS_DIR / 'capacity.json'} in {time.time() - t_start:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
