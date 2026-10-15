"""Intuition: The EDA notebooks were never run in this clone, so this one-time script builds the
processed artifacts they would have produced - always with column selection, never a raw dense frame."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.csv as pacsv

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from feed.build import (DAY_MS, LAM, add_features, finalize,
                        k_core, load_events, read_table)  # noqa: E402

RAW = ROOT.parent / "data" / "raw" / "KuaiSAR_v2"
ART = ROOT / "artifacts"
SESSION_GAP_MS = 30 * 60 * 1000


def load_cat_map(max_rows: int | None) -> tuple[pd.Series, int]:
    types = {"item_id": pa.int32(), "first_level_category_id": pa.string()}
    it = read_table(RAW / "item_features.csv",
                    ["item_id", "first_level_category_id"], types, max_rows)
    cat = pd.to_numeric(it.pop("first_level_category_id"), errors="coerce")
    ok = cat.notna()
    codes = pd.Categorical(cat[ok]).codes
    mapping = pd.Series(codes, index=it.loc[ok, "item_id"].to_numpy(), dtype="int8")
    n_cats = int(codes.max()) + 1
    print(f"[prepare] categories: {n_cats} found, {len(mapping):,} items mapped")
    return mapping, n_cats


def _sessions_series(df: pd.DataFrame) -> pd.Series:
    prev_ts = df["ts"].shift()
    prev_u = df["user_id"].shift()
    new = (df["user_id"] != prev_u) | ((df["ts"] - prev_ts) > SESSION_GAP_MS)
    return new.cumsum()


def session_frame(df: pd.DataFrame) -> pd.DataFrame:
    feed = df[df["is_search"] == 0].sort_values(["user_id", "ts"], kind="stable")
    feed = feed.assign(session_id=_sessions_series(feed))
    sizes = feed.groupby("session_id")["click"].agg(["size", "mean"])
    feed = feed.join(sizes["size"], on="session_id").join(
        sizes["mean"].rename("eng"), on="session_id")
    return feed


def real_marginals(feed: pd.DataFrame) -> dict:
    click = feed["click"].to_numpy() == 1
    clicks = feed[click]
    hour_rate = feed.groupby("hour")["click"].mean()
    pri = feed["ui_prior_n"].clip(upper=3)
    repeat = feed.assign(pri=pri).groupby("pri")["click"].mean()
    lens = feed["size"]
    out = {
        "feed_click_rate": float(click.mean()),
        "like_given_click": float((clicks["like"] == 1).mean()),
        "follow_given_click": float((clicks["follow"] == 1).mean()),
        "forward_given_click": float((clicks["forward"] == 1).mean()),
        "longview_given_click": float((clicks["play_ratio"] >= 0.7).mean()),
        "hour_click_rate": {str(int(h)): float(v) for h, v in hour_rate.items()},
        "repeat_click_rate": {str(int(k)): float(v) for k, v in repeat.items()},
        "median_session_items": float(np.median(lens)) if lens is not None else None,
        "session_hours": None,
        "engagement_by_len": None,
    }
    if lens is not None:
        buckets = {"1": lens <= 1, "2-3": (lens >= 2) & (lens <= 3), "4-10": (lens >= 4) & (lens <= 10),
                   "11-30": (lens >= 11) & (lens <= 30), "31-100": (lens >= 31) & (lens <= 100),
                   "100+": lens >= 101}
        out["engagement_by_len"] = {k: float(feed.loc[m, "click"].mean())
                                    for k, m in buckets.items() if m.any()}
    starts = feed.loc[feed.groupby("session_id")["ts"].idxmin(), "hour"]
    out["session_hours"] = {str(int(h)): float(v) for h, v in
                            starts.value_counts(normalize=True).sort_index().items()}
    return out


def user_aggregates(df: pd.DataFrame, n_cats: int, feed: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    tend = df.groupby("user_id")["ts"].transform("max")
    w = np.exp(LAM * (df["ts"] - tend).to_numpy() / DAY_MS)
    d = df.assign(w=w * df["score"].to_numpy())
    uc = d.groupby(["user_id", "cat_id"], sort=False)["w"].sum().rename("decay")
    u = d.groupby("user_id", sort=False)["w"].sum().rename("udecay")
    cnt = d.groupby(["user_id", "cat_id"], sort=False).size().rename("n")
    un = d.groupby("user_id", sort=False).size().rename("un")
    sm = d.groupby(["user_id", "cat_id"], sort=False)["score"].mean().rename("score_mean")
    interest = pd.concat([uc, cnt, sm], axis=1).reset_index()
    interest["decay_share"] = (interest["decay"] / interest.groupby("user_id")["decay"].transform("sum")).astype("float32")
    interest["exposure_share"] = (interest["n"] / interest["user_id"].map(un)).astype("float32")
    interest["lift"] = (interest["decay_share"] / (interest["exposure_share"] + 1e-9)).astype("float32")
    interest["decay"] = interest["decay"].astype("float32")
    interest["score_mean"] = interest["score_mean"].astype("float32")

    g = df.groupby("user_id", sort=False)
    ustat = pd.DataFrame({
        "n_events": g.size(),
        "n_days": g["day"].nunique(),
        "mean_score": g["score"].mean(),
        "pos_rate": g["pos"].mean(),
        "rate_like": g["like"].mean(),
        "rate_follow": g["follow"].mean(),
        "rate_forward": g["forward"].mean(),
        "n_cats": g["cat_id"].nunique(),
    })
    ustat["events_per_day"] = ustat["n_events"] / ustat["n_days"].clip(lower=1)
    fs = feed.groupby("user_id", sort=False)["session_id"].nunique().rename("n_sessions")
    fl = feed.groupby("user_id", sort=False)["size"].median().rename("median_session_items")
    ustat = ustat.join(fs).join(fl)
    ustat["sessions_per_day"] = (ustat["n_sessions"] / ustat["n_days"].clip(lower=1)).fillna(1.0)
    ustat["median_session_items"] = ustat["median_session_items"].fillna(1.0)
    ustat = ustat.reset_index()
    return interest, ustat


def item_aggregates(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby("item_id", sort=False)
    out = pd.DataFrame({
        "cat_id": g["cat_id"].first(),
        "n": g.size(),
        "mean_score": g["score"].mean(),
        "pos_rate": g["pos"].mean(),
        "play_ratio": g["play_ratio"].mean(),
        "duration_ms": g["duration_ms"].median(),
    }).reset_index()
    out["play_ratio"] = out["play_ratio"].fillna(0.0)
    out["duration_ms"] = out["duration_ms"].fillna(0.0)
    return out


def write_dictionary(n_cats: int) -> None:
    rows = []
    groups = {"u_": ("user habit", "1"), "uc_": ("user-category affinity", "1"),
              "i_": ("item quality", "1"), "ui_": ("user-item memory", "1"),
              "cold_": ("cold-start fallback", "1"), "ctx_": ("context", "1"),
              "cat_id": ("category", "1")}
    desc = {
        "u_pos_rate": "prior positive share of user events",
        "u_mean_score": "prior mean additive intent score",
        "u_rate_click": "prior click rate", "u_rate_like": "prior like rate",
        "u_rate_follow": "prior follow rate", "u_rate_forward": "prior forward rate",
        "u_play_ratio": "prior mean observed play ratio",
        "u_n_events": "prior event count", "u_recency": "days since prior event (capped 7)",
        "uc_decay_share": "decayed intent share of category for user",
        "uc_lift": "decayed share / exposure share", "uc_exposure_share": "prior category share",
        "uc_score": "prior mean intent score in category", "uc_repeat": "prior exposures to this item",
        "uc_recency": "days since prior event in category (capped 7)",
        "i_pos_rate": "prior positive share of item impressions",
        "i_mean_score": "prior mean intent score for item",
        "i_play_ratio": "prior mean observed play ratio for item",
        "i_duration": "item median duration (ms, static)",
        "i_n": "prior impression count (model uses log1p)",
        "ui_prior_n": "prior exposures of this item by this user",
        "ui_prior_max_tier": "max tier this user had on this item before",
        "cold_user": "user has zero prior events", "cold_item": "item has zero prior impressions",
        "cold_uc": "no prior events in this user-category",
        "ctx_hour_sin": "sin of UTC+8 hour", "ctx_hour_cos": "cos of UTC+8 hour",
        "cat_id": f"first-level category id 0..{n_cats - 1} (+{n_cats}=unknown)",
    }
    for name, d in desc.items():
        grp = next((v[0] for k, v in groups.items() if name.startswith(k)), "label")
        rows.append({"feature": name, "group": grp, "description": d, "live_online": True})
    for name in ("tier", "score", "pos", "watch", "obs_like", "obs_follow", "obs_forward", "obs_watch"):
        rows.append({"feature": name, "group": "label/mask", "description": "offline label", "live_online": False})
    pd.DataFrame(rows).to_csv(ART / "feature_dictionary.csv", index=False)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", action="store_true", help="tiny run for smoke testing")
    ap.add_argument("--max-rows", type=int, default=None)
    args = ap.parse_args()
    max_rows = args.max_rows if args.max_rows else (300_000 if args.fast else None)
    t0 = time.time()
    ART.mkdir(exist_ok=True)
    print("[prepare] raw headers feed:", [c for c in pacsv.read_csv(
        str(RAW / "rec_inter.csv")).column_names])
    cat_map, n_cats = load_cat_map(1_000_000 if args.fast else None)
    df = load_events(RAW, max_rows)
    df = k_core(df)

    t0_feat = time.time()
    df = add_features(df, cat_map, n_cats)
    feed_sess = session_frame(df)
    interest, ustat = user_aggregates(df, n_cats, feed_sess)
    item_stats = item_aggregates(df)
    df = finalize(df)
    print(f"[prepare] features + aggregates in {time.time() - t0_feat:.1f}s")

    cuts = np.quantile(df["ts"].to_numpy(), [0.50, 0.70, 0.85]).astype("int64")
    masks = {
        "train": (df["ts"] >= cuts[0]) & (df["ts"] < cuts[1]),
        "val": (df["ts"] >= cuts[1]) & (df["ts"] < cuts[2]),
        "test": df["ts"] >= cuts[2],
    }
    for name, m in masks.items():
        df[m].to_parquet(ART / f"features_{name}.parquet", compression="zstd", index=False)
        print(f"[prepare] wrote features_{name}.parquet rows={int(m.sum()):,}")

    interest.to_parquet(ART / "user_interest.parquet", index=False)
    ustat.to_parquet(ART / "user_stats.parquet", index=False)
    item_stats.to_parquet(ART / "item_stats.parquet", index=False)

    marginals = real_marginals(feed_sess)
    with open(ART / "real_marginals.json", "w") as f:
        json.dump(marginals, f, indent=1)
    with open(ART / "session_hours.json", "w") as f:
        json.dump(marginals["session_hours"], f, indent=1)

    config = {
        "seed": 42, "tier_gains": [0, 1, 2, 3, 4],
        "intent_weights": {"click": 1, "long_view": 2, "like": 3, "forward": 4, "follow": 5,
                           "search_click_bonus": 2, "hate": -3},
        "long_view_threshold": 0.7, "half_life_days": 3.0, "lookback_days": 7.0,
        "k_core": {"user": 5, "item": 3}, "split_cuts": [int(c) for c in cuts],
        "split_ratios": [0.50, 0.70, 0.85], "n_cats": n_cats,
        "unknown_cat": n_cats, "cat_embed_rows": n_cats + 1,
        "rows": {k: int(v.sum()) for k, v in masks.items()},
    }
    with open(ART / "preprocess_config.json", "w") as f:
        json.dump(config, f, indent=1)
    schema = {
        "columns": {c: str(df[c].dtype) for c in df.columns},
        "observed_by_source": {
            "feed": ["click", "like", "follow", "forward", "watch", "play_ratio"],
            "search": ["click"], "note": "unlogged actions are 0 in features, masked in loss",
        },
    }
    with open(ART / "schema.json", "w") as f:
        json.dump(schema, f, indent=1)
    write_dictionary(n_cats)
    print(f"[prepare] full pipeline done in {time.time() - t0:.1f}s")
    print(f"[prepare] marginals: click={marginals['feed_click_rate']:.3f} "
          f"like|click={marginals['like_given_click']:.4f} "
          f"follow|click={marginals['follow_given_click']:.4f} "
          f"median_sess={marginals['median_session_items']}")


if __name__ == "__main__":
    main()
