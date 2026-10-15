"""Intuition: The live, incremental feature store the serving loop keeps — decayed per-user and
per-user-category counters, seen-item memory and live item statistics — emitting exactly the 28
features the offline Taste model was trained on. Parity is checked by replaying real events.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
ART = ROOT / "artifacts"
DAY_MS = 86_400_000.0
HOUR_MS = 3_600_000.0
TZ_MS = 8 * HOUR_MS
LAM_MS = float(np.log(2.0) / (3.0 * DAY_MS))
LOOKBACK = 7.0

with open(ART / "feature_list.json") as _f:
    META = json.load(_f)
DENSE = META["dense"]
CAT_COL = META["cat_col"]
N_CATS = int(META["cat_embed_rows"])
ITEM_COLS = ["i_pos_rate", "i_mean_score", "i_play_ratio", "i_duration", "i_n"]


def hour_sincos(ts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    h = (ts // HOUR_MS) % 24.0
    a = 2.0 * np.pi * h / 24.0
    return np.sin(a).astype(np.float32), np.cos(a).astype(np.float32)


class FeatureState:
    """Mutable per-bot / per-item statistics; `pair_features` mirrors the offline feature list."""

    def __init__(self, catalog, n_bots: int) -> None:
        c = catalog
        self.catalog = c
        self.n_items, self.n_cats, self.n_bots = c.n_items, c.n_cats, n_bots
        z = np.zeros
        self.b_n = z(n_bots, np.int64)
        for a in ("score", "pos", "click", "like", "follow", "forward", "play", "n_play"):
            setattr(self, f"b_sum_{a}", z(n_bots, np.float64))
        self.b_last_ts = z(n_bots, np.float64)
        self.b_decay = z(n_bots, np.float64)
        self.b_decay_ref = z(n_bots, np.float64)
        sh = (n_bots, c.n_cats)
        self.uc_n = z(sh, np.int64)
        self.uc_sum_score = z(sh, np.float64)
        self.uc_decay = z(sh, np.float64)
        self.uc_ref = z(sh, np.float64)
        self.uc_last_ts = z(sh, np.float64)
        self.ui: dict[int, list] = {}
        self.item_n = c.init_n.astype(np.float64).copy()
        self.item_sum_score = c.init_sum_score.astype(np.float64).copy()
        self.item_sum_pos = c.init_sum_pos.astype(np.float64).copy()
        self.item_sum_play = c.init_sum_play.astype(np.float64).copy()
        self.item_n_play = c.init_n_play.astype(np.float64).copy()
        self.pop_sum = c.pop_sum.astype(np.float64).copy()
        self.pop_count = c.pop_count.astype(np.float64).copy()

    def reset_items(self, idx: np.ndarray) -> None:
        for arr in (self.item_n, self.item_sum_score, self.item_sum_pos,
                    self.item_sum_play, self.item_n_play, self.pop_sum, self.pop_count):
            arr[idx] = 0.0

    def item_mean_score(self) -> np.ndarray:
        return self.item_sum_score / np.maximum(self.item_n, 1.0)

    def item_popularity(self) -> np.ndarray:
        return self.pop_sum / np.maximum(self.pop_count, 1e-9)

    def cold_items(self, cold_max_n: int) -> np.ndarray:
        return self.item_n <= cold_max_n

    def update(self, bs: np.ndarray, its: np.ndarray, ts: np.ndarray, tier: np.ndarray,
               outcome: dict[str, np.ndarray]) -> None:
        """Apply one step of impressions (bots unique per step; items may repeat)."""
        c = self.catalog
        cat = c.cat[its].astype(np.int64)
        score = outcome["score"]
        f = np.exp(-LAM_MS * (ts - self.b_decay_ref[bs]))
        self.b_decay[bs] = self.b_decay[bs] * f + score
        self.b_decay_ref[bs] = ts
        self.b_n[bs] += 1
        for a in ("score", "pos", "click", "like", "follow", "forward"):
            getattr(self, f"b_sum_{a}")[bs] += outcome[a]
        self.b_sum_play[bs] += outcome["watch"]
        self.b_sum_n_play[bs] += outcome["observed_play"]
        self.b_last_ts[bs] = ts
        uidx = (bs, cat)
        fu = np.exp(-LAM_MS * (ts - self.uc_ref[uidx]))
        self.uc_decay[uidx] = self.uc_decay[uidx] * fu + score
        self.uc_ref[uidx] = ts
        self.uc_n[uidx] += 1
        self.uc_sum_score[uidx] += score
        self.uc_last_ts[uidx] = ts
        np.add.at(self.item_n, its, 1.0)
        np.add.at(self.item_sum_score, its, score)
        np.add.at(self.item_sum_pos, its, outcome["pos"])
        np.add.at(self.item_sum_play, its, outcome["watch"])
        np.add.at(self.item_n_play, its, outcome["observed_play"])
        np.add.at(self.pop_sum, its, score)
        np.add.at(self.pop_count, its, 1.0)
        for b, i, t in zip(bs.tolist(), its.tolist(), tier.tolist()):
            k = b * self.n_items + i
            e = self.ui.get(k)
            if e is None:
                self.ui[k] = [1, int(t)]
            else:
                e[0] += 1
                if t > e[1]:
                    e[1] = int(t)

    def uc_stats(self, bs: np.ndarray, its: np.ndarray, ts: np.ndarray) -> tuple:
        """Lightweight (exposure_share, decay_share, lift) for ranking without the item block."""
        cat = self.catalog.cat[its].astype(np.int64)
        bs = bs.astype(np.int64)
        uc_n = self.uc_n[bs, cat].astype(np.float64)
        uc_decay = self.uc_decay[bs, cat] * np.exp(-LAM_MS * (ts - self.uc_ref[bs, cat]))
        b_decay = self.b_decay[bs] * np.exp(-LAM_MS * (ts - self.b_decay_ref[bs]))
        exp_share = uc_n / np.maximum(self.b_n[bs], 1.0)
        decay_share = uc_decay / np.maximum(b_decay, 1e-12)
        lift = decay_share / (exp_share + 1e-9)
        return exp_share, decay_share, lift

    def seen_counts(self, bs: np.ndarray, its: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        n = np.zeros(len(bs), np.int32)
        mx = np.zeros(len(bs), np.int8)
        ui = self.ui
        for j, (b, i) in enumerate(zip(bs.tolist(), its.tolist())):
            e = ui.get(b * self.n_items + i)
            if e is not None:
                n[j], mx[j] = e[0], e[1]
        return n, mx

    def pair_features(self, bs: np.ndarray, its: np.ndarray, ts: np.ndarray
                      ) -> tuple[np.ndarray, np.ndarray]:
        """(n, 27) raw dense features in feature_list order + cat ids, exactly as offline."""
        c = self.catalog
        bs = bs.astype(np.int64)
        its = its.astype(np.int64)
        cat = c.cat[its].astype(np.int64)
        un = self.b_n[bs].astype(np.float64)
        us = np.maximum(un, 1.0)
        uc_n = self.uc_n[bs, cat].astype(np.float64)
        uc_decay = self.uc_decay[bs, cat] * np.exp(-LAM_MS * (ts - self.uc_ref[bs, cat]))
        b_decay = self.b_decay[bs] * np.exp(-LAM_MS * (ts - self.b_decay_ref[bs]))
        exp_share = uc_n / us
        decay_share = uc_decay / np.maximum(b_decay, 1e-12)
        i_n = self.item_n[its]
        ins = np.maximum(i_n, 1.0)
        ui_n, ui_max = self.seen_counts(bs, its)
        hsin, hcos = hour_sincos(ts.astype(np.float64))
        cols = {
            "u_pos_rate": self.b_sum_pos[bs] / us,
            "u_mean_score": self.b_sum_score[bs] / us,
            "u_rate_click": self.b_sum_click[bs] / us,
            "u_rate_like": self.b_sum_like[bs] / us,
            "u_rate_follow": self.b_sum_follow[bs] / us,
            "u_rate_forward": self.b_sum_forward[bs] / us,
            "u_play_ratio": self.b_sum_play[bs] / np.maximum(self.b_sum_n_play[bs], 1.0),
            "u_n_events": un,
            "u_recency": np.where(un > 0, np.clip((ts - self.b_last_ts[bs]) / DAY_MS, 0, LOOKBACK),
                                  LOOKBACK),
            "uc_decay_share": decay_share,
            "uc_lift": decay_share / (exp_share + 1e-9),
            "uc_exposure_share": exp_share,
            "uc_score": self.uc_sum_score[bs, cat] / np.maximum(uc_n, 1.0),
            "uc_repeat": ui_n.astype(np.float64),
            "uc_recency": np.where(uc_n > 0, np.clip((ts - self.uc_last_ts[bs, cat]) / DAY_MS, 0,
                                                     LOOKBACK), LOOKBACK),
            "i_pos_rate": self.item_sum_pos[its] / ins,
            "i_mean_score": self.item_sum_score[its] / ins,
            "i_play_ratio": self.item_sum_play[its] / np.maximum(self.item_n_play[its], 1.0),
            "i_duration": c.duration[its].astype(np.float64),
            "i_n": i_n,
            "ui_prior_n": ui_n.astype(np.float64),
            "ui_prior_max_tier": ui_max.astype(np.float64),
            "cold_user": (un == 0).astype(np.float64),
            "cold_item": (i_n == 0).astype(np.float64),
            "cold_uc": (uc_n == 0).astype(np.float64),
            "ctx_hour_sin": hsin.astype(np.float64),
            "ctx_hour_cos": hcos.astype(np.float64),
        }
        x = np.empty((len(its), len(DENSE)), dtype=np.float32)
        for j, name in enumerate(DENSE):
            x[:, j] = cols[name]
        return x, cat.astype(np.int64)


def _replay_user(user_id: int, events: list[dict], test_ts: set) -> list[dict]:
    """Pure-python independent replay of one user's as-of USER-side features (27-5 item cols)."""
    st = {"n": 0, "score": 0.0, "pos": 0.0, "click": 0.0, "like": 0.0, "follow": 0.0,
          "forward": 0.0, "play": 0.0, "nplay": 0.0, "last": None, "decay": 0.0, "ref": None,
          "cat": {}, "item": {}}
    out = []
    for e in events:
        t = e["ts"]
        cn = st["cat"].get(e["cat"])
        ucn = cn["n"] if cn else 0
        ucdecay = cn["decay"] * np.exp(-LAM_MS * (t - cn["ref"])) if cn else 0.0
        bdecay = st["decay"] * np.exp(-LAM_MS * (t - st["ref"])) if st["ref"] is not None else 0.0
        n = st["n"]
        ns = max(n, 1)
        exp_share = ucn / ns
        dshare = ucdecay / max(bdecay, 1e-12)
        rec = min((t - st["last"]) / DAY_MS, LOOKBACK) if st["last"] is not None else LOOKBACK
        ucrec = min((t - cn["last"]) / DAY_MS, LOOKBACK) if cn and cn["n"] else LOOKBACK
        it = st["item"].get(e["item"], [0, 0])
        h = (((t + TZ_MS) // HOUR_MS) % 24) * 2 * np.pi / 24
        if t in test_ts:
            out.append({
                "user_id": user_id, "item_id": e["item"], "ts": int(t),
                "u_pos_rate": st["pos"] / ns, "u_mean_score": st["score"] / ns,
                "u_rate_click": st["click"] / ns, "u_rate_like": st["like"] / ns,
                "u_rate_follow": st["follow"] / ns, "u_rate_forward": st["forward"] / ns,
                "u_play_ratio": st["play"] / max(st["nplay"], 1.0), "u_n_events": float(n),
                "u_recency": rec, "uc_decay_share": dshare,
                "uc_lift": dshare / (exp_share + 1e-9), "uc_exposure_share": exp_share,
                "uc_score": (cn["score"] / ucn if cn and ucn else 0.0), "uc_repeat": float(it[0]),
                "uc_recency": ucrec, "ui_prior_n": float(it[0]),
                "ui_prior_max_tier": float(it[1]), "cold_user": float(n == 0),
                "cold_uc": float(ucn == 0), "ctx_hour_sin": np.sin(h), "ctx_hour_cos": np.cos(h)})
        st["n"] = n + 1
        st["score"] += e["score"]; st["pos"] += e["pos"]; st["click"] += e["click"]
        st["like"] += e["like"]; st["follow"] += e["follow"]; st["forward"] += e["forward"]
        st["play"] += e["play"]; st["nplay"] += e["nplay"]
        st["last"] = t
        st["decay"] = bdecay + e["score"]; st["ref"] = t
        if cn is None:
            st["cat"][e["cat"]] = {"n": 1, "score": e["score"], "decay": e["score"], "ref": t,
                                   "last": t}
        else:
            cn["n"] += 1; cn["score"] += e["score"]; cn["decay"] = ucdecay + e["score"]
            cn["ref"] = t; cn["last"] = t
        item = st["item"].get(e["item"])
        if item is None:
            st["item"][e["item"]] = [1, int(e["tier"])]
        else:
            item[0] += 1; item[1] = max(item[1], int(e["tier"]))
    return out
