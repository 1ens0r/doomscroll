"""Intuition: Phase-1 entry point: train the Taste model, score val/test once, and check the
ladder B1 random < B2 popularity < B3 affinity < Taste on graded NDCG@10 with bootstrap CIs."""
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

from feed.data import (ART, ROOT, fit_scaler, load_config, load_scaler, load_split, load_xy,
                       save_scaler)
from feed.eval import (acceptance, baseline_scores, bootstrap_ci, collect_paired, lgbm_ceiling,
                       per_user_ndcg, strip_per_user)
from feed.model import block_indices, build_model, load_meta, n_params, predict_probs, score_from_probs
from feed.train import train_taste

MODELS = ROOT / "models"
POLICY_PATH = ROOT / "config" / "policy.yaml"
SCOPES = ("feed", "search", "feed_cold")


def load_policy() -> dict:
    with open(POLICY_PATH) as f:
        return yaml.safe_load(f)


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_bundle(split: str, scaler: dict) -> dict:
    d = load_xy(split, scaler)
    raw = load_split(split, columns=["i_mean_score", "uc_lift", "i_n", "uc_decay_share"])
    d["quality"] = raw["i_mean_score"].to_numpy(np.float64)
    d["lift"] = raw["uc_lift"].to_numpy(np.float64)
    d["diag_exposure"] = raw["i_n"].to_numpy(np.float64)
    d["diag_decay"] = raw["uc_decay_share"].to_numpy(np.float64)
    return d


def subsample(d: dict, n: int | None, rng: np.random.Generator) -> dict:
    if n is None or len(d["x"]) <= n:
        return d
    idx = np.sort(rng.choice(len(d["x"]), size=n, replace=False))
    return {k: v[idx] for k, v in d.items()}


def scores_for(d: dict, model: torch.nn.Module, policy: dict) -> dict[str, np.ndarray]:
    probs = predict_probs(model, d["x"], d["cat"])
    out = {"taste": score_from_probs(probs, policy["score_weights"])}
    out.update(baseline_scores(d["quality"], d["lift"], len(d["x"]), policy["seed"]))
    out["diag_exposure_pop"] = d["diag_exposure"]
    out["diag_decay_share"] = d["diag_decay"]
    return out


def scope_masks(d: dict) -> dict[str, np.ndarray]:
    feed = d["is_search"] == 0
    return {"feed": feed, "search": ~feed, "feed_cold": feed & (d["cold_item"] == 1)}


def evaluate(d: dict, scores: dict[str, np.ndarray], policy: dict, gains: list[int]) -> dict:
    ecfg = policy["eval"]
    out: dict = {}
    for scope, keep in scope_masks(d).items():
        out[scope] = {}
        for method, score in scores.items():
            summary, per_user = {}, {}
            for k in ecfg["ndcg_ks"]:
                users, vals = per_user_ndcg(score, d["tier"], d["user_id"], gains, k, keep=keep,
                                            min_impressions=ecfg["min_impressions"],
                                            min_tiers=ecfg["min_tiers"])
                summary[f"ndcg@{k}"] = bootstrap_ci(vals, ecfg["bootstrap_resamples"],
                                                    ecfg["bootstrap_seed"])
                per_user[k] = (users, vals)
            out[scope][method] = {"summary": summary, "per_user": per_user}
    return out


def print_ladder(title: str, ev: dict, ks: list[int]) -> None:
    print(f"\n== {title} ==")
    for scope in SCOPES:
        print(f"-- {scope} --")
        header = f"{'method':<14}" + "".join(f"   ndcg@{k}  [95% CI]   " for k in ks)
        print(header.rstrip())
        for method, e in ev[scope].items():
            cells = ""
            for k in ks:
                s = e["summary"][f"ndcg@{k}"]
                cells += f"   {s['mean']:.4f}  [{s['lo']:.4f}, {s['hi']:.4f}]"
            print(f"{method:<14}{cells}")


def print_paired(paired: dict, primary_k: int) -> None:
    print(f"\n== Paired bootstrap diffs (feed, ndcg@{primary_k}) ==")
    for split, methods in paired.items():
        for name, entry in methods.items():
            c = entry[f"ndcg@{primary_k}"]
            verdict = "PASS" if c["lo"] > 0 else "FAIL"
            print(f"{split:<4} {name:<16} diff={c['mean_diff']:+.4f} "
                  f"[{c['lo']:+.4f}, {c['hi']:+.4f}]  {verdict}")


def main() -> int:
    ap = argparse.ArgumentParser(description="Phase 1: Taste model + offline eval")
    ap.add_argument("--fast", action="store_true", help="tiny samples, 1 epoch, no LightGBM")
    ap.add_argument("--reuse", action="store_true",
                    help="reuse models/taste.pt (or taste_fast.pt with --fast) if it exists")
    ap.add_argument("--no-lgbm", action="store_true", help="skip the optional LightGBM ceiling")
    ap.add_argument("--boot", type=int, default=None, help="override bootstrap resamples")
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()

    t_start = time.time()
    policy = load_policy()
    seed = policy["seed"] if args.seed is None else args.seed
    seed_all(seed)
    torch.set_num_threads(int(policy["python_threads"]))
    meta = load_meta()
    gains = load_config()["tier_gains"]
    fast = args.fast
    if fast:
        policy = copy.deepcopy(policy)
        policy["train"]["max_epochs"] = 1
        policy["eval"]["bootstrap_resamples"] = min(policy["eval"]["bootstrap_resamples"], 500)
    if args.boot is not None:
        policy["eval"]["bootstrap_resamples"] = args.boot

    scaler = load_scaler() if (ART / "scaler.json").exists() else fit_scaler(load_split("train"))
    if not (ART / "scaler.json").exists():
        save_scaler(scaler)
    nn_train = 200_000 if fast else None
    nn_val = 300_000 if fast else None
    nn_test = 200_000 if fast else None
    rng = np.random.default_rng(seed)
    t0 = time.time()
    train = subsample(load_bundle("train", scaler), nn_train, rng)
    val = subsample(load_bundle("val", scaler), nn_val, rng)
    test = subsample(load_bundle("test", scaler), nn_test, rng)
    t_load = time.time() - t0
    print(f"[data] rows train={len(train['x']):,} val={len(val['x']):,} test={len(test['x']):,} "
          f"in {t_load:.1f}s")

    model = build_model(meta, policy)
    ckpt_path = MODELS / ("taste_fast.pt" if fast else "taste.pt")
    print(f"[model] TasteNet params={n_params(model):,} ({len(meta['dense'])} dense + cat)")

    if args.reuse and ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["state_dict"])
        report = {"history": ckpt.get("history", []), "best_epoch": ckpt.get("best_epoch", -1),
                  "best_val_ndcg10": ckpt.get("best_val_ndcg10", float("nan")),
                  "pos_weights": ckpt.get("pos_weights", []), "n_params": n_params(model),
                  "n_train_rows": ckpt.get("n_train_rows", 0)}
        t_train = 0.0
        print(f"[train] reused {ckpt_path}")
    else:
        t0 = time.time()
        report = train_taste(model, train, val, policy, gains, seed)
        t_train = time.time() - t0
        MODELS.mkdir(exist_ok=True)
        torch.save({"state_dict": model.state_dict(), "blocks": block_indices(meta["dense"]),
                    "cat_rows": int(meta["cat_embed_rows"]), "model_cfg": policy["model"],
                    "feature_list": meta, "pos_weights": report["pos_weights"],
                    "best_epoch": report["best_epoch"],
                    "best_val_ndcg10": report["best_val_ndcg10"],
                    "history": report["history"], "n_train_rows": report["n_train_rows"],
                    "seed": seed}, ckpt_path)
        print(f"[train] best epoch {report['best_epoch']} val ndcg@10={report['best_val_ndcg10']:.4f}"
              f" in {t_train:.1f}s")

    t0 = time.time()
    scores_val = scores_for(val, model, policy)
    scores_test = scores_for(test, model, policy)
    t_pred = time.time() - t0
    t0 = time.time()
    ev_val = evaluate(val, scores_val, policy, gains)
    ev_test = evaluate(test, scores_test, policy, gains)
    t_eval = time.time() - t0

    lgbm_out: dict = {"status": "skipped"}
    lgbm_ev: dict[str, dict] = {}
    if not fast and not args.no_lgbm:
        try:
            lgbm_out, lgbm_scores = lgbm_ceiling(train, {"val": val, "test": test}, policy,
                                                 gains, seed)
            for split_name, sc in lgbm_scores.items():
                d = val if split_name == "val" else test
                lgbm_ev[split_name] = evaluate(d, sc, policy, gains)
            print(f"[lgbm] ceiling {lgbm_out['status']} in {lgbm_out['seconds']:.0f}s "
                  f"(best iter {lgbm_out['best_iteration']})")
        except Exception as exc:  # ceiling is optional; log, never fail the phase
            lgbm_out = {"status": f"failed: {type(exc).__name__}: {exc}"}
            print(f"[lgbm] skipped: {exc}")

    pairs = [("taste", "b3_affinity", "taste_minus_b3"),
             ("b3_affinity", "b2_popularity", "b3_minus_b2"),
             ("b2_popularity", "b1_random", "b2_minus_b1")]
    ks = policy["eval"]["ndcg_ks"]
    paired = collect_paired({"val": ev_val, "test": ev_test}, pairs, ks, policy["eval"]["bootstrap_seed"])
    acc = acceptance(paired, policy["eval"]["primary_k"])

    print_ladder("LADDER val", ev_val, ks)
    print_ladder("LADDER test", ev_test, ks)
    print_paired(paired, policy["eval"]["primary_k"])
    if lgbm_ev:
        print_ladder("CEILING (LightGBM lambdarank) val", lgbm_ev["val"], ks)
    gate_note = " (smoke only; not enforced)" if fast else ""
    print(f"\nACCEPTANCE ({acc['metric']}, feed, val+test): "
          f"{'PASS' if acc['pass'] else 'FAIL'}{gate_note}")

    timings = {"load": t_load, "train": t_train, "predict": t_pred, "eval": t_eval,
               "total": time.time() - t_start}
    print("[timing] " + "  ".join(f"{k}={v:.1f}s" for k, v in timings.items()))
    results = {
        "phase": 1, "fast": fast, "seed": seed, "reused": args.reuse,
        "rows": {s: int(len(b["x"])) for s, b in (("train", train), ("val", val), ("test", test))},
        "gains": gains,
        "model": {k: report[k] for k in ("n_params", "best_epoch", "best_val_ndcg10",
                                         "pos_weights", "history")},
        "timings_sec": timings,
        "ladder": {"val": strip_per_user(ev_val), "test": strip_per_user(ev_test)},
        "paired": paired,
        "acceptance": acc,
        "lgbm_ceiling": {**lgbm_out, "ladder": {s: strip_per_user(e) for s, e in lgbm_ev.items()}},
    }
    suffix = "_fast" if fast else ""
    out_path = ART / f"offline_results{suffix}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=1)
    print(f"[done] results -> {out_path}")
    return 0 if (fast or acc["pass"]) else 1


if __name__ == "__main__":
    sys.exit(main())
