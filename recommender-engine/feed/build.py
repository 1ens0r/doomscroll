"""Intuition: Turn the raw KuaiSAR event CSVs into one clean, candidate-level table where every
feature is computed strictly from what happened *before* the event, so offline labels stay honest."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.csv as pacsv

HALF_LIFE_DAYS = 3.0
LOOKBACK_DAYS = 7.0
LONG_VIEW = 0.7
USER_MIN, ITEM_MIN = 5, 3
DAY_MS = 86_400_000.0
TZ_MS = 8 * 3_600_000.0
LAM = float(np.log(2.0) / HALF_LIFE_DAYS)

FEED_COLS = ["user_id", "item_id", "duration_ms", "playing_time", "timestamp",
             "forward", "like", "follow", "click"]
FEED_TYPES = {"user_id": pa.int32(), "item_id": pa.int32(), "duration_ms": pa.float32(),
              "playing_time": pa.float32(), "timestamp": pa.float64(), "forward": pa.int32(),
              "like": pa.int32(), "follow": pa.int32(), "click": pa.int32()}
SRC_COLS = ["user_id", "item_id", "search_session_timestamp", "click_cnt"]
SRC_TYPES = {"user_id": pa.int32(), "item_id": pa.int32(),
             "search_session_timestamp": pa.float64(), "click_cnt": pa.int32()}


def read_table(path: Path, cols: list[str], types: dict[str, pa.DataType],
               max_rows: int | None = None) -> pd.DataFrame:
    opts = pacsv.ConvertOptions(include_columns=cols, column_types=types)
    if max_rows is None:
        return pacsv.read_csv(str(path), convert_options=opts).to_pandas()
    reader = pacsv.open_csv(str(path), convert_options=opts)
    batches, n = [], 0
    for batch in reader:
        batches.append(batch)
        n += batch.num_rows
        if n >= max_rows:
            break
    return pa.Table.from_batches(batches).slice(0, max_rows).to_pandas()


def load_events(raw: Path, max_rows: int | None = None) -> pd.DataFrame:
    """Read feed + search logs with column selection, clean, dedup, and attach labels."""
    feed = read_table(raw / "rec_inter.csv", FEED_COLS, FEED_TYPES, max_rows)
    feed = feed.rename(columns={"timestamp": "ts", "playing_time": "playing"})
    feed["is_search"] = np.int8(0)
    src = read_table(raw / "src_inter.csv", SRC_COLS, SRC_TYPES, max_rows)
    src = src.rename(columns={"search_session_timestamp": "ts"})
    src["click"] = (src.pop("click_cnt") > 0).astype("int8")
    for c in ("forward", "like", "follow"):
        src[c] = np.int8(0)
    src["duration_ms"] = np.float32(np.nan)
    src["playing"] = np.float32(np.nan)
    src["is_search"] = np.int8(1)
    df = pd.concat([feed, src], ignore_index=True)
    df["ts"] = df["ts"].astype("int64")
    n0 = len(df)
    df = df.drop_duplicates(subset=["user_id", "item_id", "ts", "is_search"], keep="first")
    dropped = n0 - len(df)
    for c in ("click", "like", "follow", "forward"):
        df[c] = df[c].clip(0, 1).astype("int8")
    bad = ~(df["duration_ms"] > 0)
    df.loc[bad, "duration_ms"] = np.float32(np.nan)
    df["play_ratio"] = (df["playing"] / df["duration_ms"]).clip(0.0, 5.0).astype("float32")
    df = df.drop(columns=["playing"])
    _add_labels(df)
    print(f"[build] events={len(df):,} (dedup dropped {dropped:,}); "
          f"feed={int((df['is_search'] == 0).sum()):,} search={int(df['is_search'].sum()):,}")
    return df


def _add_labels(df: pd.DataFrame) -> None:
    """tier 0..4, additive intent score, pos mask - see ASSUMPTIONS A4.
    tier2 is search-click or an *unclicked* long view (a clicked long view is still tier1):
    this reproduces the README's tier mix (tier1 ~43%, tier2 ~2%)."""
    click = df["click"].to_numpy() == 1
    long_view = df["play_ratio"].to_numpy() >= LONG_VIEW
    like = df["like"].to_numpy() == 1
    follow = df["follow"].to_numpy() == 1
    forward = df["forward"].to_numpy() == 1
    is_search = df["is_search"].to_numpy() == 1
    tier = np.zeros(len(df), dtype=np.int8)
    tier[click] = 1
    tier[long_view & ~click] = 2
    tier[like] = 3
    tier[follow | forward] = 4
    tier[is_search & click] = np.maximum(tier[is_search & click], 2)
    score = (click.astype(np.int8) + 2 * long_view.astype(np.int8) + 3 * like.astype(np.int8)
             + 4 * forward.astype(np.int8) + 5 * follow.astype(np.int8)
             + 2 * (is_search & click).astype(np.int8))
    df["tier"] = tier
    df["score"] = score.astype(np.int8)
    df["pos"] = (score > 0).astype(np.int8)


def k_core(df: pd.DataFrame, user_min: int = USER_MIN, item_min: int = ITEM_MIN) -> pd.DataFrame:
    """Iteratively drop users/items with too few interactions (README 5/3 core)."""
    it = 0
    for it in range(20):
        uc = df["user_id"].value_counts()
        ic = df["item_id"].value_counts()
        keep = (df["user_id"].map(uc).ge(user_min).to_numpy()
                & df["item_id"].map(ic).ge(item_min).to_numpy())
        if keep.all():
            break
        df = df[keep]
    print(f"[build] k-core ({it} iters): {len(df):,} rows, {df['user_id'].nunique():,} users, "
          f"{df['item_id'].nunique():,} items")
    return df


def _safe_div(a: pd.Series | np.ndarray, b: pd.Series | np.ndarray) -> pd.Series | np.ndarray:
    if isinstance(a, pd.Series):
        return (a / b.where(b > 0)).fillna(0.0)
    return np.divide(a, b, out=np.zeros(np.broadcast(a, b).shape, dtype=np.float64),
                     where=np.asarray(b) > 0)


def _prior_sum(df: pd.DataFrame, gcol: str, col: str) -> pd.Series:
    return df.groupby(gcol, sort=False)[col].cumsum() - df[col]


def _item_prior_features(item: np.ndarray, ts: np.ndarray, pos: np.ndarray,
                         score: np.ndarray, play: np.ndarray,
                         obs: np.ndarray) -> tuple[np.ndarray, ...]:
    """As-of item features under a true global time order: sort by (item, ts) once,
    take exclusive cumulative sums per item, scatter back to the original row order."""
    n = len(item)
    order = np.lexsort((ts, item))
    it = item[order]
    starts = np.r_[0, np.flatnonzero(it[1:] != it[:-1]) + 1]
    counts = np.diff(np.r_[starts, n])
    rep = np.repeat(starts, counts)

    def prior(v: np.ndarray) -> np.ndarray:
        vv = v[order]
        cs = np.cumsum(vv)
        base = np.repeat(cs[starts] - vv[starts], counts)
        ps = cs - base - vv
        out = np.empty(n, dtype=np.float64)
        out[order] = ps
        return out

    ones = np.ones(n, dtype=np.float64)
    pn = prior(ones)
    pscore = prior(score.astype(np.float64))
    ppos = prior(pos.astype(np.float64))
    pplay = prior(np.nan_to_num(play, nan=0.0).astype(np.float64))
    pobs = prior(obs.astype(np.float64))
    i_n = pn.astype(np.int32)
    i_mean = _safe_div(pscore, pn)
    i_pos = _safe_div(ppos, pn)
    i_play = _safe_div(pplay, pobs)
    return i_n, i_pos, i_mean, i_play


def _key_prior_stats(key: np.ndarray, tier: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Prior count and prior max tier per key: one stable sort, no huge groupby."""
    n = len(key)
    order = np.argsort(key, kind="stable")
    sk = key[order]
    starts = np.r_[0, np.flatnonzero(sk[1:] != sk[:-1]) + 1]
    counts = np.diff(np.r_[starts, n])
    prior_n = np.arange(n, dtype=np.int64) - np.repeat(starts, counts)
    t = tier[order].astype(np.int32)
    max_prior = np.zeros(n, dtype=np.int8)
    for k in (1, 2, 3, 4):
        v = (t >= k).astype(np.int32)
        cs = np.cumsum(v)
        prior_ge = cs - np.repeat(cs[starts] - v[starts], counts) - v > 0
        max_prior += prior_ge.astype(np.int8)
    out_n = np.empty(n, dtype=np.int32)
    out_max = np.empty(n, dtype=np.int8)
    out_n[order] = prior_n.astype(np.int32)
    out_max[order] = max_prior
    return out_n, out_max


def add_features(df: pd.DataFrame, cat_map: pd.Series, n_cats: int) -> pd.DataFrame:
    """Every feature here is as-of the current event: strictly prior history only."""
    df = df.sort_values(["user_id", "ts"], kind="stable", ignore_index=True)
    df["cat_id"] = df["item_id"].map(cat_map).fillna(n_cats).astype("int8")
    df["obs_play"] = df["play_ratio"].notna().astype("int8")
    u = "user_id"
    df["u_n_events"] = df.groupby(u, sort=False).cumcount().astype("int32")
    df["u_pos_rate"] = _safe_div(_prior_sum(df, u, "pos"), df["u_n_events"])
    df["u_mean_score"] = _safe_div(_prior_sum(df, u, "score"), df["u_n_events"])
    for a in ("click", "like", "follow", "forward"):
        df[f"u_rate_{a}"] = _safe_div(_prior_sum(df, u, a), df["u_n_events"])
    df["u_play_ratio"] = _safe_div(_prior_sum(df, u, "play_ratio"), _prior_sum(df, u, "obs_play"))
    rec = df.groupby(u, sort=False)["ts"].diff() / DAY_MS
    df["u_recency"] = rec.fillna(LOOKBACK_DAYS).clip(upper=LOOKBACK_DAYS).astype("float32")
    dt = (df["ts"] - df["ts"].min()) / DAY_MS
    weighted = df["score"].to_numpy("float64") * np.exp(LAM * dt.to_numpy())
    df["_w"] = weighted
    df["_udecay"] = _prior_sum(df, u, "_w").to_numpy() * np.exp(-LAM * dt.to_numpy())
    ucode = pd.factorize(df["user_id"].astype("int64") * (n_cats + 1) + df["cat_id"])[0]
    df["_ucode"] = ucode
    uc_n = df.groupby("_ucode", sort=False).cumcount()
    df["uc_decay"] = (_prior_sum(df, "_ucode", "_w").to_numpy()
                      * np.exp(-LAM * dt.to_numpy())).astype("float32")
    df["uc_score"] = _safe_div(_prior_sum(df, "_ucode", "score"), uc_n)
    df["uc_exposure_share"] = _safe_div(uc_n, df["u_n_events"])
    df["uc_decay_share"] = _safe_div(df["uc_decay"], df["_udecay"]).astype("float32")
    df["uc_lift"] = _safe_div(df["uc_decay_share"], df["uc_exposure_share"] + 1e-9).astype("float32")
    ucrec = df.groupby("_ucode", sort=False)["ts"].diff() / DAY_MS
    df["uc_recency"] = ucrec.fillna(LOOKBACK_DAYS).clip(upper=LOOKBACK_DAYS).astype("float32")
    i = "item_id"
    i_n, i_pos, i_mean, i_play = _item_prior_features(
        df[i].to_numpy(), df["ts"].to_numpy(), df["pos"].to_numpy(),
        df["score"].to_numpy(), df["play_ratio"].to_numpy(), df["obs_play"].to_numpy())
    df["i_n"] = i_n
    df["i_pos_rate"] = i_pos.astype("float32")
    df["i_mean_score"] = i_mean.astype("float32")
    df["i_play_ratio"] = i_play.astype("float32")
    dur_med = df.groupby(i, sort=False)["duration_ms"].median()
    df["i_duration"] = df[i].map(dur_med).fillna(0.0).astype("float32")
    key = df["user_id"].astype("int64") * 10_000_000 + df["item_id"]
    ui_n, ui_max = _key_prior_stats(key.to_numpy(), df["tier"].to_numpy())
    df["ui_prior_n"] = ui_n
    df["ui_prior_max_tier"] = ui_max
    df["uc_repeat"] = ui_n
    df["cold_user"] = (df["u_n_events"] == 0).astype("int8")
    df["cold_item"] = (df["i_n"] == 0).astype("int8")
    df["cold_uc"] = (uc_n == 0).astype("int8")
    hour = ((df["ts"] + TZ_MS) // 3_600_000) % 24
    df["ctx_hour_sin"] = np.sin(2 * np.pi * hour / 24).astype("float32")
    df["ctx_hour_cos"] = np.cos(2 * np.pi * hour / 24).astype("float32")
    df["hour"] = hour.astype("int8")
    day = ((df["ts"] + TZ_MS) // DAY_MS).astype("int64")
    df["day"] = (day - day.min()).astype("int16")
    drop_tmp = ["_w", "_udecay", "_ucode", "obs_play"]
    return df.drop(columns=drop_tmp)


FEATURE_COLS = [
    "u_pos_rate", "u_mean_score", "u_rate_click", "u_rate_like", "u_rate_follow",
    "u_rate_forward", "u_play_ratio", "u_n_events", "u_recency",
    "uc_decay_share", "uc_lift", "uc_exposure_share", "uc_score", "uc_repeat", "uc_recency",
    "i_pos_rate", "i_mean_score", "i_play_ratio", "i_duration", "i_n",
    "ui_prior_n", "ui_prior_max_tier", "cold_user", "cold_item", "cold_uc",
    "ctx_hour_sin", "ctx_hour_cos", "cat_id",
]
LABEL_COLS = ["user_id", "item_id", "ts", "day", "hour", "tier", "score", "pos", "is_search"]
MASK_COLS = ["obs_like", "obs_follow", "obs_forward", "obs_watch"]


def finalize(df: pd.DataFrame) -> pd.DataFrame:
    """Attach observation masks and keep only the columns downstream code needs."""
    feed = (df["is_search"] == 0).astype("int8")
    df["obs_like"] = feed
    df["obs_follow"] = feed
    df["obs_forward"] = feed
    df["obs_watch"] = feed
    df["watch"] = df["play_ratio"].clip(0.0, 1.0)
    return df[LABEL_COLS + FEATURE_COLS + MASK_COLS + ["watch"]]
