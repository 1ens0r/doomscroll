"""Intuition: Entry point for the Learn stage — generate simulator logs, train the Taste ranker on
them, and save a deployment checkpoint whose inputs match the live serving path exactly.
"""
from __future__ import annotations

import argparse
import json
import random
import time

import numpy as np
import torch
import yaml

from feed.data import ROOT, load_config as load_preprocess_config, load_scaler
from feed.learning import generate_logs, make_bundle
from feed.model import HEADS, block_indices, build_model, load_meta, n_params, predict_probs
from feed.train import train_taste
from feed.world import load_config

MODELS = ROOT / "models"
POLICY_PATH = ROOT / "config" / "policy.yaml"
ART = ROOT / "artifacts"


def load_policy() -> dict:
    with open(POLICY_PATH) as f:
        return yaml.safe_load(f)


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def main() -> int:
    ap = argparse.ArgumentParser(description="Phase 2.5: learn the ranker from simulator logs")
    ap.add_argument("--fast", action="store_true", help="small swarm, 1 epoch")
    ap.add_argument("--policy", default="personal_affinity",
                    help="behaviour policy whose logs train the ranker")
    ap.add_argument("--bots", type=int, default=None)
    ap.add_argument("--days", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--policy-config", action="store_true")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--ckpt", default="taste_sim.pt")
    ap.add_argument("--variant", default=None, help="model variant (e.g. wide); default baseline")
    ap.add_argument("--loss", default=None, help="train loss name (e.g. hybrid); default from yaml")
    args = ap.parse_args()

    t_start = time.time()
    policy = load_policy()
    seed = int(args.seed if args.seed is not None else policy["seed"])
    seed_all(seed)
    torch.set_num_threads(int(policy["python_threads"]))
    cfg = load_config()
    cfg["policy"] = policy
    if args.fast:
        policy["train"]["max_epochs"] = 1

    meta = load_meta()
    gains = load_preprocess_config()["tier_gains"]
    scaler = load_scaler()
    if args.variant:
        policy["_model_variant"] = args.variant
    if args.loss:
        policy["train"]["loss"] = args.loss

    n_bots = args.bots if args.bots else (1000 if args.fast else 3000)
    n_days = args.days if args.days else (3 if args.fast else 5)
    if args.epochs is not None:
        policy["train"]["max_epochs"] = args.epochs

    t0 = time.time()
    raw = generate_logs(cfg, args.policy, seed, n_bots, n_days)
    train, val = make_bundle(raw, scaler, val_frac=0.2, seed=seed)
    t_gen = time.time() - t0

    variant = policy.get("_model_variant")
    policy["_gains"] = list(gains)
    model = build_model(meta, policy, variant=variant)
    print(f"[learn] TasteNet params={n_params(model):,} variant={variant or 'baseline'} "
          f"loss={policy['train'].get('loss', 'masked_bce')}")
    t0 = time.time()
    report = train_taste(model, train, val, policy, gains, seed)
    t_train = time.time() - t0

    probs = predict_probs(model, val["x"], val["cat"])
    weights = np.array([policy["score_weights"][h] for h in HEADS], dtype=np.float32)
    scores = probs @ weights
    from feed.eval import per_user_ndcg
    _, nd = per_user_ndcg(scores, val["tier"], val["user_id"], gains,
                          policy["eval"]["primary_k"],
                          min_impressions=policy["eval"]["min_impressions"],
                          min_tiers=policy["eval"]["min_tiers"])
    val_ndcg = float(nd.mean()) if len(nd) else float("nan")

    MODELS.mkdir(exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "variant": variant,
                "blocks": block_indices(meta["dense"]),
                "cat_rows": int(meta["cat_embed_rows"]), "model_cfg": policy["model"],
                "feature_list": meta, "pos_weights": report["pos_weights"],
                "best_epoch": report["best_epoch"], "best_val_ndcg10": report["best_val_ndcg10"],
                "history": report["history"], "n_train_rows": report["n_train_rows"],
                "source": f"sim-logs:{args.policy}", "seed": seed}, MODELS / args.ckpt)
    print(f"[learn] best epoch {report['best_epoch']} val ndcg@{policy['eval']['primary_k']}="
          f"{val_ndcg:.4f} | saved {args.ckpt} | gen {t_gen:.0f}s train {t_train:.0f}s")

    summary = {
        "policy": args.policy, "bots": n_bots, "days": n_days, "seed": seed,
        "rows": {"train": report["n_train_rows"], "val": int(len(val["x"]))},
        "model": {"n_params": report["n_params"], "best_epoch": report["best_epoch"],
                  "best_val_ndcg10": report["best_val_ndcg10"], "history": report["history"]},
        "val_ndcg": val_ndcg, "timings_sec": {"generate": t_gen, "train": t_train,
                                              "total": time.time() - t_start},
        "checkpoint": args.ckpt,
    }
    with open(ART / "learn_results.json", "w") as f:
        json.dump(summary, f, indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
