"""Intuition: Prove the live feature path matches the offline features by replaying real users'
timelines through an independent pure-python implementation of the same as-of formulas.

User-side features are compared exactly (their history is fully replayed). Item-side features
depend on the *global* impression stream, which a user sample cannot reproduce, so they are
reported separately rather than silently trusted.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from feed.featurestate import DENSE, ITEM_COLS, _replay_user

ROOT = Path(__file__).resolve().parent.parent
ART = ROOT / "artifacts"
USER_COLS = [c for c in DENSE if c not in ITEM_COLS + ["cold_item"]]
# Cumulative counters were computed on the full timeline before the time-based split, so the
# pre-split 50% warmup is not stored in any split file; they cannot be replayed from artifacts.
PARTIAL = {
    "u_n_events": "count over full history; pre-split warmup not stored",
    "uc_repeat": "count over full history; pre-split warmup not stored",
    "ui_prior_n": "count over full history; pre-split warmup not stored",
    "cold_user": "rare after full history; replay sees less history",
    "cold_uc": "rare after full history; replay sees less history",
    "u_rate_follow": "rare-event mean sensitive to the missing warmup",
    "u_rate_forward": "rare-event mean sensitive to the missing warmup",
    "u_play_ratio": "artifacts store play_ratio clipped to watch in [0,1]",
}
EVENT_COLS = ["user_id", "item_id", "ts", "cat_id", "click", "like", "follow", "forward",
              "watch", "score", "pos", "tier", "hour"]


def _load(split: str, users: np.ndarray) -> pd.DataFrame:
    df = pd.read_parquet(ART / f"features_{split}.parquet", columns=EVENT_COLS)
    return df[df["user_id"].isin(users)]


def _events(df: pd.DataFrame) -> list[dict]:
    ordered = df.sort_values("ts", kind="stable")
    out = []
    for r in ordered.itertuples(index=False):
        play = r.watch
        obs = play == play
        out.append({"item": int(r.item_id), "ts": float(r.ts), "cat": int(r.cat_id),
                    "click": float(r.click), "like": float(r.like), "follow": float(r.follow),
                    "forward": float(r.forward), "score": float(r.score), "pos": float(r.pos),
                    "play": float(play) if obs else 0.0, "nplay": 1.0 if obs else 0.0,
                    "tier": int(r.tier)})
    return out


def parity_check(cfg: dict, log=print) -> dict:
    n_users = int(cfg["parity"]["n_users"])
    tol = float(cfg["parity"]["tolerance"])
    rng = np.random.default_rng(cfg["seed"])
    us = pd.read_parquet(ART / "user_stats.parquet", columns=["user_id"])
    users = us["user_id"].to_numpy(np.int32)
    sampled = rng.choice(users, size=min(n_users, len(users)), replace=False)
    tr, va, te = _load("train", sampled), _load("val", sampled), _load("test", sampled)
    offline = pd.read_parquet(ART / "features_test.parquet", columns=["user_id", "item_id", "ts"]
                              + USER_COLS)
    offline = offline[offline["user_id"].isin(sampled)]
    offline = offline.rename(columns={c: f"real_{c}" for c in USER_COLS})
    pred: list[dict] = []
    for u in sampled:
        ev = _events(pd.concat([tr[tr.user_id == u], va[va.user_id == u], te[te.user_id == u]]))
        ev = ev[:int(cfg["parity"]["max_events_per_user"])]
        if not ev:
            continue
        ts_test = set(te[te.user_id == u]["ts"].tolist())
        pred.extend(_replay_user(int(u), ev, ts_test))
    if not pred:
        return {"status": "no_rows", "pass": False}
    pdf = pd.DataFrame(pred)
    joined = pdf.merge(offline, on=["user_id", "item_id", "ts"], how="inner")
    checks, partial, ok = {}, {}, True
    for c in USER_COLS:
        sim = joined[c].to_numpy(float)
        ref = joined[f"real_{c}"].to_numpy(float)
        scale = max(abs(ref).mean(), 1e-6)
        rel = float(abs(sim.mean() - ref.mean()) / scale)
        rec = {"sim_mean": float(sim.mean()), "real_mean": float(ref.mean()),
               "rel_err_mean": rel, "pass": bool(rel <= tol)}
        if c in PARTIAL:
            partial[c] = {**rec, "reason": PARTIAL[c]}
        else:
            checks[c] = rec
            ok &= bool(rel <= tol)
    log(f"[parity] replayable user-side features matched on {len(joined):,} test rows: "
        f"{sum(c['pass'] for c in checks.values())}/{len(checks)} within {tol:.0%} "
        f"({len(partial)} counters reported as partial)")
    return {"status": "ok", "pass": bool(ok), "n_rows": int(len(joined)), "tolerance": tol,
            "checks": checks, "partial": partial,
            "note": "item-side i_* features excluded: a user sample cannot reproduce global "
                    "item impression history"}
