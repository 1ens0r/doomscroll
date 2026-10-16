"""Intuition: Phase A driver. One process trains the reference (baseline + masked BCE) and every
candidate (bigger network and/or the hybrid ranking-aware objective) on identical data, split, seed,
and candidate protocol; selection is on validation NDCG@10 with paired-bootstrap CIs; the winner and
the reference are then touched on test exactly once."""
from __future__ import annotations

import argparse
import copy
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
from feed.model import load_meta

POLICY_PATH = ROOT / "config" / "policy.yaml"
RESULTS_DIR = ROOT.parent / "notes" / "results"

# (label, model variant, loss name, extra policy overrides)
CONFIGS: list[tuple[str, str | None, str, dict]] = [
    ("baseline_masked", None, "masked_bce", {}),
    ("wide_masked", "wide", "masked_bce", {}),
    ("wide_hybrid", "wide", "hybrid", {}),
    ("big_hybrid", "big", "hybrid", {}),
    ("big_hybrid_r1", "big", "hybrid", {"train.rank_weight": 1.0}),
]
REFERENCE = "baseline_masked"


def load_policy() -> dict:
    with open(POLICY_PATH) as f:
        return yaml.safe_load(f)


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def apply_overrides(policy: dict, overrides: dict) -> dict:
    out = copy.deepcopy(policy)
    for dotted, value in overrides.items():
        section, _, key = dotted.partition(".")
        out[section][key] = value
    return out


def ci_of(a: dict, b: dict, k: int, n_boot: int, seed: int) -> dict:
    ua, va = a["scopes"]["val"]["feed"]["taste"]["per_user"][str(k)]
    ub, vb = b["scopes"]["val"]["feed"]["taste"]["per_user"][str(k)]
    if not np.array_equal(ua, ub):
        raise RuntimeError("paired user mismatch")
    return paired_bootstrap(va, vb, n_boot, seed)


def print_table(results: list[dict], ci: dict, primary: int) -> None:
    print(f"\n== Phase A model/loss (val, feed, ndcg@{primary}) ==")
    print(f"{'config':<18}{'params':>10}{'best_ep':>8}{'val@10':>9}{'val@20':>9}{'v_loss':>9}"
          f"   diff vs baseline [95% CI]")
    for r in results:
        v10 = r["scopes"]["val"]["feed"]["taste"][f"ndcg@{primary}"]["mean"]
        v20 = r["scopes"]["val"]["feed"]["taste"]["ndcg@20"]["mean"]
        vl = r["history"][-1]["val_loss"]
        c = ci.get(r["label"])
        ctxt = (f"{c['mean_diff']:+.4f} [{c['lo']:+.4f}, {c['hi']:+.4f}]" if c else "(reference)")
        print(f"{r['label']:<18}{r['n_params']:>10,}{r['best_epoch']:>8}{v10:>9.4f}{v20:>9.4f}"
              f"{vl:>9.4f}   {ctxt}")


def main() -> int:
    ap = argparse.ArgumentParser(description="Phase A: improve network + objective")
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--only", default="", help="comma labels to run")
    ap.add_argument("--test", default="", help="comma labels to retrain and evaluate on test once")
    ap.add_argument("--max-rows", type=int, default=None)
    ap.add_argument("--boot", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None)
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

    if args.test:
        wanted = [n for n in args.test.split(",") if n]
    elif args.only:
        wanted = [n for n in args.only.split(",") if n]
    else:
        wanted = [c[0] for c in CONFIGS]
    configs = [c for c in CONFIGS if c[0] in wanted]
    # Always keep the reference present for paired comparisons.
    if args.test and not any(c[0] == REFERENCE for c in configs):
        configs = [c for c in CONFIGS if c[0] == REFERENCE] + configs

    bundles = load_bundles(policy, max_rows=max_rows, seed=seed)
    print(f"[data] rows train={len(bundles['train']['x']):,} val={len(bundles['val']['x']):,} "
          f"test={len(bundles['test']['x']):,}")

    results = []
    for label, variant, loss, overrides in configs:
        pol = apply_overrides(policy, overrides)
        pol["train"]["loss"] = loss
        r = run_one(bundles, pol, meta, gains, label, variant=variant,
                    evaluate_test=bool(args.test), seed=seed,
                    ckpt_dir=ROOT / "models" / "improve")
        r["label"] = label
        r["model_variant"] = variant or "baseline"
        print(f"[{label}] variant={variant or 'baseline'} loss={loss} best_epoch={r['best_epoch']} "
              f"val@10={r['best_val_ndcg10']:.4f} train={r['train_seconds']:.1f}s")
        results.append(r)

    n_boot = policy["eval"]["bootstrap_resamples"]
    ref = next((r for r in results if r["label"] == REFERENCE), None)
    ci = {r["label"]: ci_of(r, ref, primary, n_boot, seed) for r in results if ref and r is not ref}
    print_table(results, ci, primary)

    best = max(results, key=lambda r: r["scopes"]["val"]["feed"]["taste"][f"ndcg@{primary}"]["mean"])
    within = []
    for r in results:
        if r is best:
            within.append(r)
            continue
        c = ci_of(r, best, primary, n_boot, seed)
        if c["lo"] <= 0 <= c["hi"]:
            within.append(r)
    promoted = min(within, key=lambda r: (r["n_params"], r["train_seconds"])) if within else best
    print(f"\n[promote] best val = {best['label']} "
          f"({best['scopes']['val']['feed']['taste'][f'ndcg@{primary}']['mean']:.4f}); "
          f"within-CI = {[r['label'] for r in within]} -> cheapest = {promoted['label']}")

    payload: dict = {
        "phase": "A", "fast": args.fast, "seed": seed, "max_rows": max_rows, "primary_k": primary,
        "reference": REFERENCE,
        "configs": {label: {"variant": variant, "loss": loss, "overrides": dict(overrides)}
                    for label, variant, loss, overrides in configs},
        "results": {r["label"]: {"n_params": r["n_params"], "best_epoch": r["best_epoch"],
                                 "history": r["history"], "train_seconds": r["train_seconds"],
                                 "peak_rss_mb": r["peak_rss_mb"],
                                 "val_feed": {f"ndcg@{k}": r["scopes"]["val"]["feed"]["taste"][f"ndcg@{k}"]
                                              for k in policy["eval"]["ndcg_ks"]}}
                    for r in results},
        "paired_vs_reference": ci,
        "best": best["label"], "within_ci": [r["label"] for r in within],
        "promoted": promoted["label"],
    }
    if args.test:
        payload["test_feed"] = {r["label"]: {"ndcg@10": r["scopes"]["test"]["feed"]["taste"]["ndcg@10"],
                                             "ndcg@20": r["scopes"]["test"]["feed"]["taste"]["ndcg@20"]}
                                for r in results}
    ART.mkdir(exist_ok=True)
    out = ART / ("improve_results_fast.json" if args.fast else "improve_results.json")
    out.write_text(json.dumps(payload, indent=1))
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "improve.json").write_text(json.dumps(payload, indent=1))
    print(f"[done] {out} + {RESULTS_DIR / 'improve.json'} in {time.time() - t_start:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
