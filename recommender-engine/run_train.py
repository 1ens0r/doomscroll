"""Intuition: the final long training run for the one promoted network. Unlike the ablation drivers,
this trains a single `wide` model for many epochs (OneCycle/cosine), evaluates it on validation every
epoch and on test exactly once, saves the checkpoint under `weights/`, and writes a full per-epoch
history to `notes/results/train_<tag>.json` for the notebook's training curves.

    P=~/miniconda3/envs/picograd/bin/python
    $P run_train.py --variant wide --loss hybrid      --epochs 22 --patience 5 --tag wide_hybrid
    $P run_train.py --variant wide --loss lambdarank  --epochs 22 --patience 5 --tag wide_rank
"""
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

from feed.data import ROOT, load_config
from feed.eval import paired_bootstrap
from feed.experiment import load_bundles, run_one
from feed.model import load_meta, n_params

POLICY_PATH = ROOT / "config" / "policy.yaml"
RESULTS_DIR = ROOT.parent / "notes" / "results"
WEIGHTS_DIR = ROOT / "weights"


def main() -> int:
    ap = argparse.ArgumentParser(description="Long final training run for the wide network")
    ap.add_argument("--variant", default="wide")
    ap.add_argument("--loss", default="hybrid")
    ap.add_argument("--epochs", type=int, default=22)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--lr", type=float, default=0.002)
    ap.add_argument("--scheduler", default="onecycle")
    ap.add_argument("--rank-weight", type=float, default=None)
    ap.add_argument("--dropout", type=float, default=None)
    ap.add_argument("--tag", default="wide")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--test", action="store_true")
    ap.add_argument("--max-rows", type=int, default=None)
    args = ap.parse_args()

    t_start = time.time()
    policy = yaml.safe_load(POLICY_PATH.read_text())
    torch.set_num_threads(int(args.threads))
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    policy["train"]["max_epochs"] = args.epochs
    policy["train"]["early_stop_patience"] = args.patience
    policy["train"]["lr"] = args.lr
    policy["train"]["scheduler"] = args.scheduler
    policy["train"]["loss"] = args.loss
    if args.rank_weight is not None:
        policy["train"]["rank_weight"] = args.rank_weight
    if args.dropout is not None:
        for v in policy["model"]["variants"]:
            if v["name"] == args.variant:
                v["dropout"] = args.dropout

    meta = load_meta()
    gains = load_config()["tier_gains"]
    bundles = load_bundles(policy, max_rows=args.max_rows, seed=args.seed)
    print(f"[data] train={len(bundles['train']['x']):,} val={len(bundles['val']['x']):,} "
          f"test={len(bundles['test']['x']):,}  threads={args.threads}")

    name = f"{args.tag}"
    r = run_one(bundles, policy, meta, gains, name, variant=args.variant,
                evaluate_test=bool(args.test), seed=args.seed,
                ckpt_dir=ROOT / "models" / "final")
    print(f"[{name}] params={r['n_params']:,} best_epoch={r['best_epoch']} "
          f"val@10={r['scopes']['val']['feed']['taste']['ndcg@10']['mean']:.4f} "
          f"train={r['train_seconds']:.1f}s")

    val = r["scopes"]["val"]["feed"]
    test = r["scopes"].get("test", {}).get("feed", {})
    methods = ["taste", "b1_random", "b2_popularity", "b3_affinity"]

    def ndcg(scope: dict, method: str, k: int) -> dict:
        return scope[method][f"ndcg@{k}"] if method in scope else {}

    paired_pop = {}
    for split, scope in (("val", val), ("test", test)):
        if scope and "taste" in scope:
            ua, va = scope["taste"]["per_user"]["10"]
            ub, vb = scope["b2_popularity"]["per_user"]["10"]
            if np.array_equal(ua, ub):
                paired_pop[split] = paired_bootstrap(va, vb,
                                                     policy["eval"]["bootstrap_resamples"],
                                                     policy["eval"]["bootstrap_seed"])

    payload = {
        "phase": "final", "tag": name, "variant": args.variant, "loss": args.loss,
        "n_params": r["n_params"],
        "max_epochs": args.epochs, "patience": args.patience, "lr": args.lr,
        "scheduler": args.scheduler, "rank_weight": policy["train"].get("rank_weight"),
        "dropout": policy["model"].get("dropout"), "seed": args.seed,
        "best_epoch": r["best_epoch"], "best_val_ndcg10": r["best_val_ndcg10"],
        "train_seconds": r["train_seconds"], "peak_rss_mb": r["peak_rss_mb"],
        "pos_weights": r["pos_weights"], "history": r["history"],
        "val_feed": {m: {f"ndcg@{k}": ndcg(val, m, k) for k in policy["eval"]["ndcg_ks"]}
                     for m in methods},
        "test_feed": {m: {f"ndcg@{k}": ndcg(test, m, k) for k in policy["eval"]["ndcg_ks"]}
                      for m in methods} if test else {},
        "paired_vs_popularity": paired_pop,
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / f"train_{name}.json").write_text(json.dumps(payload, indent=1))

    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    ckpt_path = WEIGHTS_DIR / f"{name}_offline.pt"
    saved = torch.load(r["ckpt"], map_location="cpu")
    torch.save({"state_dict": saved["state_dict"], "variant": args.variant, "name": name,
                "model_cfg": policy["model"], "feature_list": meta,
                "history": r["history"], "best_epoch": r["best_epoch"]}, ckpt_path)
    print(f"[done] {RESULTS_DIR / f'train_{name}.json'} + {ckpt_path} "
          f"in {time.time() - t_start:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
