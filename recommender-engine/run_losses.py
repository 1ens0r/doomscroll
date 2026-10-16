"""Intuition: Phase 8 loss ablation. Every arm keeps the identical data, split, candidate protocol,
seed, and metric; only the objective (L0-L7) or a single optimisation knob changes. Selection is on
validation NDCG@10 with paired-bootstrap CIs; test is touched once, after selection."""
from __future__ import annotations

import argparse
import copy
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from feed.data import ART, ROOT, load_config
from feed.eval import paired_bootstrap
from feed.experiment import load_bundles, run_one
from feed.losses import brier, ece, log_loss, reliability
from feed.model import HEADS, load_meta, predict_probs, score_from_probs
from feed.train import compute_pos_weights, masked_head_loss

POLICY_PATH = ROOT / "config" / "policy.yaml"
RESULTS_DIR = ROOT.parent / "notes" / "results"
FIG_DIR = ROOT.parent / "eda" / "figures"

# name -> dotted-path overrides applied to the base policy (one factor per L5 arm).
ARMS: dict[str, dict] = {
    "L0_masked_bce": {},
    "L2_cap5": {"model.pos_weight_cap": 5.0},
    "L2_cap20": {"model.pos_weight_cap": 20.0},
    "L2_no_weighting": {"model.pos_weight_mode": "none"},
    "L2_enn": {"model.pos_weight_mode": "enn"},
    "L3_lambdarank": {"train.loss": "lambdarank"},
    "L4_equal_heads": {"train.loss": "equal_heads"},
    "L4_uncertainty": {"train.loss": "uncertainty"},
    "L5_lr_5e-3": {"train.lr": 0.005},
    "L5_lr_1e-3": {"train.lr": 0.001},
    "L5_cosine": {"train.scheduler": "cosine"},
    "L5_wd_1e-3": {"train.weight_decay": 0.001},
    "L5_dropout_0.2": {"model.dropout": 0.2},
    "L5_grad_clip_1": {"train.grad_clip": 1.0},
    "L5_patience_3": {"train.early_stop_patience": 3},
}
FAST_ARMS = ["L0_masked_bce", "L2_cap5", "L4_equal_heads", "L5_lr_5e-3"]


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


def ci_of(a: dict, b: dict, k: int, n_boot: int, seed: int, key: str = "taste") -> dict:
    ua, va = a["scopes"]["val"]["feed"][key]["per_user"][str(k)]
    ub, vb = b["scopes"]["val"]["feed"][key]["per_user"][str(k)]
    if not np.array_equal(ua, ub):
        raise RuntimeError("paired user mismatch")
    return paired_bootstrap(va, vb, n_boot, seed)


def head_grad_norms(model, policy: dict, bundle: dict, sample: int = 65536,
                    seed: int = 0) -> list[float]:
    """Gradient norm of the whole model for each isolated head loss on a fixed train sample."""
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(len(bundle["x"]), size=min(sample, len(bundle["x"])), replace=False))
    x = torch.from_numpy(np.ascontiguousarray(bundle["x"][idx]))
    c = torch.from_numpy(bundle["cat"][idx].astype(np.int64))
    y = torch.from_numpy(bundle["y"][idx])
    m = torch.from_numpy(bundle["m"][idx])
    pw = torch.tensor(compute_pos_weights(bundle["y"], bundle["m"],
                                          policy["model"]["pos_weight_cap"],
                                          policy["model"].get("pos_weight_mode", "sqrt")))
    model.eval()
    out = []
    for j in range(len(HEADS)):
        model.zero_grad(set_to_none=True)
        logits = model(x, c)
        bce = torch.nn.functional.binary_cross_entropy_with_logits(logits, y, pos_weight=pw,
                                                                   reduction="none")
        head = (bce[:, j] * m[:, j]).sum() / m[:, j].sum().clamp(min=1.0)
        head.backward()
        norm = float(torch.sqrt(sum((p.grad ** 2).sum() for p in model.parameters()
                                    if p.grad is not None)))
        out.append(norm)
    model.zero_grad(set_to_none=True)
    return out


def diagnostics(result: dict, bundles: dict, policy: dict, gains: list[int]) -> dict:
    """L1 + L7: per-head prevalence/masks/losses, gradient norms, and click/watch calibration."""
    model = result["model"]
    pol = result["policy"]
    tr, va = bundles["train"], bundles["val"]
    pw = compute_pos_weights(tr["y"], tr["m"], pol["model"]["pos_weight_cap"],
                             pol["model"].get("pos_weight_mode", "sqrt"))
    probs = predict_probs(model, va["x"], va["cat"])
    heads: dict = {}
    for j, h in enumerate(HEADS):
        obs = va["m"][:, j] > 0
        heads[h] = {
            "val_bce": float(np.mean(
                -((va["y"][obs, j] * np.log(np.clip(probs[obs, j], 1e-9, 1))) +
                  ((1 - va["y"][obs, j]) * np.log(np.clip(1 - probs[obs, j], 1e-9, 1)))))),
            "val_prevalence": float(va["y"][obs, j].mean()) if obs.any() else None,
            "val_observed": int(obs.sum()), "val_mask_rate": float(1 - obs.mean()),
            "train_prevalence": float(tr["y"][tr["m"][:, j] > 0, j].mean()),
            "brier": brier(probs[:, j], va["y"][:, j], va["m"][:, j]),
            "log_loss": log_loss(probs[:, j], va["y"][:, j], va["m"][:, j]),
            "ece": ece(probs[:, j], va["y"][:, j], va["m"][:, j]),
            "grad_norm": result["grad_norms"][j],
        }
    return {"pos_weights": [float(x) for x in pw], "pos_weight_mode": result["pos_weight_mode"],
            "heads": heads,
            "reliability": {h: reliability(probs[:, j], va["y"][:, j], va["m"][:, j])
                            for j, h in enumerate(HEADS) if h in ("click", "watch")}}


def plot_diagnostics(diag: dict, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(11, 4.2))
    for h, color in (("click", "#2b8cbe"), ("watch", "#e6550d")):
        rel = diag["reliability"][h]
        ax0.plot([0, 1], [0, 1], "--", color="0.6", lw=1)
        ax0.plot(rel["confidence"], rel["accuracy"], "o-", color=color,
                 label=f"{h} (ECE {diag['heads'][h]['ece']:.3f})")
    ax0.set_xlabel("predicted probability"); ax0.set_ylabel("observed frequency")
    ax0.set_title("L7 calibration (val, reliability)"); ax0.legend(frameon=False); ax0.grid(alpha=.3)
    heads = list(diag["heads"])
    ax1.bar(heads, [diag["heads"][h]["val_bce"] for h in heads], color="#6a51a3")
    ax1.set_ylabel("masked BCE (val)")
    ax1.set_title("per-head validation loss"); ax1.grid(alpha=.3, axis="y")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def print_table(results: list[dict], ci: dict, primary: int) -> None:
    print(f"\n== Loss ablation (val, feed, ndcg@{primary}) ==")
    print(f"{'arm':<18}{'params':>9}{'best_ep':>8}{'val@10':>9}{'val@20':>9}{'v_loss':>9}"
          f"   diff vs L0 [95% CI]")
    for r in results:
        v10 = r["scopes"]["val"]["feed"]["taste"][f"ndcg@{primary}"]["mean"]
        v20 = r["scopes"]["val"]["feed"]["taste"]["ndcg@20"]["mean"]
        vl = r["history"][-1]["val_loss"]
        c = ci.get(r["name"])
        ctxt = (f"{c['mean_diff']:+.4f} [{c['lo']:+.4f}, {c['hi']:+.4f}]" if c else "(reference)")
        print(f"{r['name']:<18}{r['n_params']:>9,}{r['best_epoch']:>8}{v10:>9.4f}{v20:>9.4f}"
              f"{vl:>9.4f}   {ctxt}")


def main() -> int:
    ap = argparse.ArgumentParser(description="Phase 8: loss ablation")
    ap.add_argument("--fast", action="store_true", help="1 epoch, 200k rows, subset of arms")
    ap.add_argument("--arms", default="", help="comma subset of arm names")
    ap.add_argument("--test", default="", help="comma arm names to retrain and evaluate on test once")
    ap.add_argument("--max-rows", type=int, default=None)
    ap.add_argument("--boot", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--no-diagnostics", action="store_true")
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
    default = FAST_ARMS if args.fast else list(ARMS)
    names = [n for n in args.arms.split(",") if n] or default
    if args.test:
        names = [n for n in args.test.split(",") if n]

    bundles = load_bundles(policy, max_rows=max_rows, seed=seed)
    print(f"[data] rows train={len(bundles['train']['x']):,} val={len(bundles['val']['x']):,} "
          f"test={len(bundles['test']['x']):,}")

    results = []
    for name in names:
        if name not in ARMS:
            raise SystemExit(f"unknown arm {name!r}; known: {list(ARMS)}")
        pol = apply_overrides(policy, ARMS[name])
        r = run_one(bundles, pol, meta, gains, name, variant=None,
                    evaluate_test=bool(args.test), seed=seed,
                    return_model=not args.no_diagnostics)
        r["overrides"] = ARMS[name]
        if not args.no_diagnostics:
            r["grad_norms"] = head_grad_norms(r["model"], pol, bundles["train"])
        print(f"[{name}] best_epoch={r['best_epoch']} val@10={r['best_val_ndcg10']:.4f} "
              f"train={r['train_seconds']:.1f}s")
        results.append(r)

    n_boot = policy["eval"]["bootstrap_resamples"]
    base = next((r for r in results if r["name"] == "L0_masked_bce"), results[0])
    ci = {r["name"]: ci_of(r, base, primary, n_boot, seed)
          for r in results if r["name"] != base["name"]}
    print_table(results, ci, primary)
    best = max(results, key=lambda r: r["scopes"]["val"]["feed"]["taste"][f"ndcg@{primary}"]["mean"])
    print(f"\n[select] best val ndcg@{primary}: {best['name']} "
          f"({best['scopes']['val']['feed']['taste'][f'ndcg@{primary}']['mean']:.4f})")

    diag = None
    if not args.no_diagnostics:
        diag = diagnostics(base, bundles, base["policy"], gains)
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        (RESULTS_DIR / "loss_diagnostics.json").write_text(json.dumps(diag, indent=1))
        try:
            plot_diagnostics(diag, FIG_DIR / "policy_06_loss_diagnostics.png")
            print(f"[diag] diagnostics + figure written")
        except Exception as exc:
            print(f"[diag] figure skipped: {exc}")

    payload = {
        "phase": 8, "fast": args.fast, "seed": seed, "max_rows": max_rows, "primary_k": primary,
        "arms": {r["name"]: {"overrides": r["overrides"], "n_params": r["n_params"],
                             "best_epoch": r["best_epoch"], "pos_weight_mode": r["pos_weight_mode"],
                             "pos_weights": r["pos_weights"], "history": r["history"],
                             "train_seconds": r["train_seconds"], "peak_rss_mb": r["peak_rss_mb"],
                             "val_feed": {f"ndcg@{k}": r["scopes"]["val"]["feed"]["taste"][f"ndcg@{k}"]
                                          for k in policy["eval"]["ndcg_ks"]}}
                 for r in results},
        "paired_vs_baseline": ci,
        "best": best["name"],
        "diagnostics_file": "notes/results/loss_diagnostics.json" if diag else None,
    }
    if args.test:
        payload["test_feed"] = {r["name"]: {"ndcg@10": r["scopes"]["test"]["feed"]["taste"]["ndcg@10"],
                                            "ndcg@20": r["scopes"]["test"]["feed"]["taste"]["ndcg@20"]}
                                for r in results}
    ART.mkdir(exist_ok=True)
    out = ART / ("loss_ablation_fast.json" if args.fast else "loss_ablation.json")
    out.write_text(json.dumps(payload, indent=1))
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "loss_ablation.json").write_text(json.dumps(
        {**{k: v for k, v in payload.items() if k != "arms"},
         "arms": {n: {kk: vv for kk, vv in a.items() if kk != "history"}
                  for n, a in payload["arms"].items()}}, indent=1))
    print(f"[done] {out} + {RESULTS_DIR / 'loss_ablation.json'} in {time.time() - t_start:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
