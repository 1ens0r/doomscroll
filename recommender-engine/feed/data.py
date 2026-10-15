"""Intuition: One place that knows how to read the processed artifacts: pick columns, standardise
dense features with TRAIN-ONLY statistics, and hand the model masked labels for its five heads."""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
ART = ROOT / "artifacts"

FEATURES = [
    "u_pos_rate", "u_mean_score", "u_rate_click", "u_rate_like", "u_rate_follow",
    "u_rate_forward", "u_play_ratio", "u_n_events", "u_recency",
    "uc_decay_share", "uc_lift", "uc_exposure_share", "uc_score", "uc_repeat", "uc_recency",
    "i_pos_rate", "i_mean_score", "i_play_ratio", "i_duration", "i_n",
    "ui_prior_n", "ui_prior_max_tier", "cold_user", "cold_item", "cold_uc",
    "ctx_hour_sin", "ctx_hour_cos", "cat_id",
]
CAT_COL = "cat_id"
DENSE = [f for f in FEATURES if f != CAT_COL]
HEADS = ["click", "like", "follow", "forward", "watch"]
HEAD_MASKS: dict[str, str | None] = {"click": None, "like": "obs_like", "follow": "obs_follow",
                                     "forward": "obs_forward", "watch": "obs_watch"}
LOG1P = ["i_n"]
ID_COLS = ["user_id", "item_id"]
EVAL_COLS = ["ts", "day", "tier", "score", "pos", "is_search", "cold_item", "cold_user"]


def load_config() -> dict:
    with open(ART / "preprocess_config.json") as f:
        return json.load(f)


def _dedup(cols: list[str]) -> list[str]:
    seen: set[str] = set()
    return [c for c in cols if not (c in seen or seen.add(c))]


def load_split(split: str, columns: list[str] | None = None) -> pd.DataFrame:
    """Column-selective parquet read; never loads all columns unless asked."""
    cols = columns or _dedup(ID_COLS + EVAL_COLS + FEATURES + HEADS + list(HEAD_MASKS.values())[1:])
    return pd.read_parquet(ART / f"features_{split}.parquet", columns=cols)


def fit_scaler(train: pd.DataFrame) -> dict:
    """Standardiser statistics from the train split only (log1p on heavy-tailed counts)."""
    x = train[DENSE].to_numpy(np.float64).copy()
    for name in LOG1P:
        x[:, DENSE.index(name)] = np.log1p(np.clip(x[:, DENSE.index(name)], 0, None))
    mean = x.mean(axis=0)
    std = np.maximum(x.std(axis=0), 1e-6)
    return {"log1p": LOG1P, "features": DENSE, "mean": mean.tolist(), "std": std.tolist(),
            "fit_on": "train", "n_rows": int(len(train))}


def standardize(df: pd.DataFrame, scaler: dict) -> np.ndarray:
    x = df[DENSE].to_numpy(np.float64)
    for name in scaler["log1p"]:
        j = DENSE.index(name)
        x[:, j] = np.log1p(np.clip(x[:, j], 0, None))
    mean = np.asarray(scaler["mean"])
    std = np.asarray(scaler["std"])
    return ((x - mean) / std).astype(np.float32)


def make_targets(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """y: five action targets; m: 1 where the action was logged, 0 where unobserved (search rows).
    Search logs clicks only, so like/follow/forward/watch are masked there (see ASSUMPTIONS A2)."""
    y = np.empty((len(df), len(HEADS)), dtype=np.float32)
    m = np.ones_like(y)
    for j, h in enumerate(HEADS):
        col = "watch" if h == "watch" else h
        y[:, j] = df[col].to_numpy(np.float32)
        mask_col = HEAD_MASKS[h]
        if mask_col is not None:
            m[:, j] = df[mask_col].to_numpy(np.float32)
    y = np.nan_to_num(y, nan=0.0)
    return y, m


def load_xy(split: str, scaler: dict) -> dict[str, np.ndarray]:
    df = load_split(split)
    y, m = make_targets(df)
    return {
        "x": standardize(df, scaler),
        "cat": df[CAT_COL].to_numpy(np.int64),
        "y": y, "m": m,
        "user_id": df["user_id"].to_numpy(np.int64),
        "tier": df["tier"].to_numpy(np.int8),
        "is_search": df["is_search"].to_numpy(np.int8),
        "cold_item": df["cold_item"].to_numpy(np.int8),
    }


def save_scaler(scaler: dict) -> None:
    with open(ART / "scaler.json", "w") as f:
        json.dump(scaler, f, indent=1)


def load_scaler() -> dict:
    with open(ART / "scaler.json") as f:
        return json.load(f)


def write_feature_list() -> None:
    cfg = load_config()
    meta = {
        "features": FEATURES, "dense": DENSE, "cat_col": CAT_COL,
        "heads": HEADS, "head_masks": HEAD_MASKS, "log1p": LOG1P,
        "n_cats": cfg["n_cats"], "cat_embed_rows": cfg["cat_embed_rows"],
        "id_cols": ID_COLS, "eval_cols": EVAL_COLS,
    }
    with open(ART / "feature_list.json", "w") as f:
        json.dump(meta, f, indent=1)


def head_rates(split: str) -> dict[str, float]:
    df = load_split(split, columns=HEADS + list(HEAD_MASKS.values())[1:])
    out = {}
    for h in HEADS:
        y = df["watch" if h == "watch" else h].to_numpy(np.float32)
        mask_col = HEAD_MASKS[h]
        m = np.ones(len(y)) if mask_col is None else df[mask_col].to_numpy()
        out[h] = float(np.nansum(y * m) / max(1.0, m.sum()))
    return out


def main() -> None:
    t0 = time.time()
    train = load_split("train")
    scaler = fit_scaler(train)
    save_scaler(scaler)
    write_feature_list()
    print(f"[data] scaler fit on train only: {scaler['n_rows']:,} rows, {len(DENSE)} dense features")
    for split in ("train", "val", "test"):
        rates = head_rates(split)
        print(f"[data] {split}: rows={len(load_split(split, columns=['tier'])):,}  " +
              "  ".join(f"{h}={r:.4f}" for h, r in rates.items()))
    print(f"[data] wrote scaler.json + feature_list.json in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
