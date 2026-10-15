"""Intuition: Graded NDCG@k over each user's label-window impressions, with per-user eligibility,
user-level bootstrap CIs, paired comparisons, and the baseline scores the Taste model must beat."""
from __future__ import annotations

import time

import numpy as np


def eligible_user_mask(users: np.ndarray, tiers: np.ndarray,
                       min_impressions: int = 2, min_tiers: int = 2
                       ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Rows of users with enough impressions AND label variance to make ranking measurable."""
    uniq, gid = np.unique(users, return_inverse=True)
    n_imp = np.bincount(gid, minlength=len(uniq))
    key = gid.astype(np.int64) * 5 + tiers.astype(np.int64)
    n_tiers = np.bincount(np.unique(key) // 5, minlength=len(uniq))
    keep_user = (n_imp >= min_impressions) & (n_tiers >= min_tiers)
    return keep_user[gid], uniq, keep_user


def _starts(sorted_group_ids: np.ndarray) -> np.ndarray:
    return np.r_[0, np.flatnonzero(sorted_group_ids[1:] != sorted_group_ids[:-1]) + 1]


def _dcg_at_k(gains_in_order: np.ndarray, starts: np.ndarray, n: int, k: int) -> np.ndarray:
    counts = np.diff(np.r_[starts, n])
    pos = np.arange(n) - np.repeat(starts, counts) + 1
    disc = np.where(pos <= k, 1.0 / np.log2(pos + 1.0), 0.0)
    return np.add.reduceat(gains_in_order * disc, starts)


def per_user_ndcg(scores: np.ndarray, tiers: np.ndarray, users: np.ndarray,
                  gains: list[int], k: int = 10, keep: np.ndarray | None = None,
                  min_impressions: int = 2, min_tiers: int = 2
                  ) -> tuple[np.ndarray, np.ndarray]:
    """NDCG@k per eligible user over the rows of one scope (caller passes keep mask per scope)."""
    scores = np.asarray(scores, dtype=np.float64)
    scores = np.where(np.isfinite(scores), scores, -1e18)
    if keep is not None:
        scores, tiers, users = scores[keep], tiers[keep], users[keep]
    rows_ok, _, _ = eligible_user_mask(users, tiers, min_impressions, min_tiers)
    s, t, u = scores[rows_ok], tiers[rows_ok], users[rows_ok]
    if len(s) == 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float64)
    gain = np.asarray(gains, dtype=np.float64)[t]
    order = np.lexsort((np.arange(len(s)), -s, u))
    su, sg = u[order], gain[order]
    starts = _starts(su)
    dcg = _dcg_at_k(sg, starts, len(su), k)
    order_ideal = np.lexsort((-sg, su))
    idcg = _dcg_at_k(sg[order_ideal], starts, len(su), k)
    ndcg = np.divide(dcg, idcg, out=np.zeros_like(dcg), where=idcg > 0)
    return su[starts], ndcg


def bootstrap_ci(values: np.ndarray, n_boot: int = 2000, seed: int = 42,
                 alpha: float = 0.05, chunk: int = 100) -> dict[str, float]:
    """Percentile CI of the user-level mean; resamples users with replacement."""
    values = np.asarray(values, dtype=np.float64)
    if len(values) == 0:
        return {"mean": float("nan"), "lo": float("nan"), "hi": float("nan"),
                "n_users": 0, "n_boot": 0}
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot, dtype=np.float64)
    done = 0
    n = len(values)
    while done < n_boot:
        c = min(chunk, n_boot - done)
        idx = rng.integers(0, n, size=(c, n))
        means[done:done + c] = values[idx].mean(axis=1)
        done += c
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"mean": float(values.mean()), "lo": float(lo), "hi": float(hi),
            "n_users": int(n), "n_boot": int(n_boot)}


def paired_bootstrap(a: np.ndarray, b: np.ndarray, n_boot: int = 2000, seed: int = 42,
                     alpha: float = 0.05, chunk: int = 100) -> dict[str, float]:
    """CI of mean(a - b) over the same users; a, b must be per-user arrays on the same users."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if len(a) == 0 or len(a) != len(b):
        return {"mean_diff": float("nan"), "lo": float("nan"), "hi": float("nan"),
                "n_users": int(len(a)), "n_boot": 0}
    rng = np.random.default_rng(seed)
    diffs = a - b
    means = np.empty(n_boot, dtype=np.float64)
    done = 0
    n = len(diffs)
    while done < n_boot:
        c = min(chunk, n_boot - done)
        idx = rng.integers(0, n, size=(c, n))
        means[done:done + c] = diffs[idx].mean(axis=1)
        done += c
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"mean_diff": float(diffs.mean()), "lo": float(lo), "hi": float(hi),
            "n_users": int(n), "n_boot": int(n_boot)}


def baseline_scores(quality: np.ndarray, lift: np.ndarray, n: int,
                    seed: int = 42) -> dict[str, np.ndarray]:
    """B1 uniform random; B2 global engagement popularity (item mean intent score); B3
    personalized popularity (quality x (1 + affinity lift), lift = decay share / exposure share)."""
    rng = np.random.default_rng(seed)
    quality = np.asarray(quality, dtype=np.float64)
    return {
        "b1_random": rng.random(n).astype(np.float64),
        "b2_popularity": quality,
        "b3_affinity": quality * (1.0 + np.asarray(lift, dtype=np.float64)),
    }


def strip_per_user(ev: dict) -> dict:
    """JSON-able view of an evaluate() result: drop the per-user arrays."""
    return {scope: {m: e["summary"] for m, e in methods.items()} for scope, methods in ev.items()}


def collect_paired(evs: dict[str, dict], pairs: list[tuple[str, str, str]],
                   ks: list[int], seed: int) -> dict:
    """Paired user-bootstrap diffs (a - b) on the feed scope for each split; users must align."""
    n_boot = evs["val"]["feed"]["taste"]["summary"][f"ndcg@{ks[0]}"]["n_boot"]
    out: dict = {}
    for split, ev in evs.items():
        feed = ev["feed"]
        out[split] = {}
        for a, b, name in pairs:
            entry = {}
            for k in ks:
                ua, va = feed[a]["per_user"][k]
                ub, vb = feed[b]["per_user"][k]
                if not np.array_equal(ua, ub):
                    raise RuntimeError(f"paired bootstrap user mismatch: {a} vs {b} @{k}")
                entry[f"ndcg@{k}"] = paired_bootstrap(va, vb, n_boot, seed)
            out[split][name] = entry
    return out


def acceptance(paired: dict, primary_k: int) -> dict:
    """Phase-1 gate: every adjacent ladder diff's 95% CI must exclude 0 on every split."""
    metric = f"ndcg@{primary_k}"
    checks, ok = {}, True
    for split, methods in paired.items():
        for name, entry in methods.items():
            ci = entry[metric]
            good = bool(ci["lo"] > 0)
            checks[f"{split}:{name}"] = {**ci, "pass": good}
            ok &= good
    return {"pass": bool(ok), "metric": metric, "checks": checks}


def lgbm_ceiling(train_b: dict, eval_bundles: dict[str, dict], policy: dict,
                 gains: list[int], seed: int) -> tuple[dict, dict[str, dict]]:
    """Optional lambdarank ceiling on identical features; never fails the run (status instead)."""
    import lightgbm as lgb

    bcfg = policy["baselines"]
    k = policy["eval"]["primary_k"]
    t0 = time.time()

    def pack(b: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        order = np.argsort(b["user_id"], kind="stable")
        x = np.hstack([b["x"][order], b["cat"][order, None].astype(np.float32)])
        y = b["tier"][order].astype(np.int32)
        gid = np.unique(b["user_id"][order], return_inverse=True)[1]
        return x, y, np.bincount(gid)

    x_tr, y_tr, g_tr = pack(train_b)
    x_va, y_va, g_va = pack(eval_bundles["val"])
    ranker = lgb.LGBMRanker(
        objective="lambdarank", label_gain=list(gains), n_estimators=bcfg["lgbm_n_estimators"],
        learning_rate=bcfg["lgbm_learning_rate"], num_leaves=bcfg["lgbm_num_leaves"],
        min_child_samples=100, subsample=0.9, subsample_freq=1, colsample_bytree=0.9,
        random_state=seed, n_jobs=policy["python_threads"], deterministic=True,
        force_row_wise=True, verbose=-1)
    ranker.fit(x_tr, y_tr, group=g_tr, eval_set=[(x_va, y_va)], eval_group=[g_va],
               eval_at=[k], categorical_feature=[x_tr.shape[1] - 1],
               callbacks=[lgb.early_stopping(bcfg["lgbm_early_stop_rounds"], verbose=False)])
    seconds = time.time() - t0
    best_iter = int(getattr(ranker, "best_iteration_", 0) or 0)
    if best_iter <= 0:
        best_iter = int(getattr(ranker.booster_, "best_iteration", 0) or 0)
    if best_iter <= 0:
        best_iter = int(ranker.booster_.current_iteration())
    status = "ok" if seconds <= bcfg["lgbm_budget_sec"] else "over_budget_but_completed"
    scores = {}
    for split, b in eval_bundles.items():
        x = np.hstack([b["x"], b["cat"][:, None].astype(np.float32)])
        scores[split] = {"lgbm_ceiling": ranker.predict(x).astype(np.float64)}
    return {"status": status, "seconds": seconds, "best_iteration": best_iter,
            "budget_sec": bcfg["lgbm_budget_sec"]}, scores
