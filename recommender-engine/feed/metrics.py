"""Intuition: One accumulator for every swarm number the plan asks for — IWE, session depth,
hourly/repeat curves for calibration, retention and the coverage/diversity guardrails.
"""
from __future__ import annotations

import numpy as np

BUCKETS = [(1, 1), (2, 3), (4, 10), (11, 30), (31, 100), (101, 10 ** 9)]
BUCKET_NAMES = ["1", "2-3", "4-10", "11-30", "31-100", "100+"]


class Metrics:
    def __init__(self, n_bots: int, n_items: int, n_cats: int, iwe: dict, sat_window: int) -> None:
        self.n_bots, self.n_items, self.n_cats, self.sat_window = n_bots, n_items, n_cats, sat_window
        self.weights = np.array([iwe["click"], iwe["long_view"], iwe["like"], iwe["forward"],
                                 iwe["follow"], iwe["hate"]], dtype=np.float64)
        self.iwe = np.zeros(n_bots, dtype=np.float64)
        self.imp = np.zeros(n_bots, dtype=np.int64)
        self.hour_imp = np.zeros(24, np.int64); self.hour_click = np.zeros(24, np.int64)
        self.rep_imp = np.zeros(4, np.int64); self.rep_click = np.zeros(4, np.int64)
        self.cat_imp = np.zeros(n_cats, np.int64); self.seen_items = np.zeros(n_items, bool)
        self.cold_imp = 0; self.cold_eng = 0; self.rep_total = 0; self.total_imp = 0
        self.total_clicks = 0; self.total_actions = np.zeros(4, np.int64)
        self.day_imp = np.zeros(n_bots, np.int64); self.day_score = np.zeros(n_bots, np.float64)
        self.day_skip = np.zeros(n_bots, np.int64); self.day_novel = np.zeros(n_bots, np.int64)
        self.sess_len: list[np.ndarray] = []
        self.sess_pos: list[np.ndarray] = []
        self.sess_hi: list[np.ndarray] = []
        self.daily_active: list[float] = []

    def step(self, bots: np.ndarray, items: np.ndarray, cats: np.ndarray, out: dict,
             prior_n: np.ndarray, cold: np.ndarray, hour: np.ndarray) -> None:
        click = out["click"].astype(np.float64)
        w = (self.weights[0] * click + self.weights[1] * out["long_view"]
             + self.weights[2] * out["like"] + self.weights[3] * out["forward"]
             + self.weights[4] * out["follow"] + self.weights[5] * out["hate"])
        np.add.at(self.iwe, bots, w)
        self.imp[bots] += 1
        np.add.at(self.hour_imp, hour, 1); np.add.at(self.hour_click, hour, click.astype(np.int64))
        rb = np.minimum(prior_n, 3)
        np.add.at(self.rep_imp, rb, 1); np.add.at(self.rep_click, rb, click.astype(np.int64))
        np.add.at(self.cat_imp, cats, 1)
        self.seen_items[items] = True
        self.cold_imp += int(cold.sum())
        self.cold_eng += int((cold & out["pos"].astype(bool)).sum())
        self.rep_total += int((prior_n > 0).sum())
        self.total_imp += len(bots)
        self.total_clicks += int(click.sum())
        self.total_actions += np.array([out["long_view"].sum(), out["like"].sum(),
                                        out["forward"].sum(), out["follow"].sum()], dtype=np.int64)
        self.day_imp[bots] += 1
        self.day_score[bots] += out["score"]
        self.day_skip[bots] += (out["tier"] == 0).astype(np.int64)
        self.day_novel[bots] += (prior_n == 0).astype(np.int64)

    def add_sessions(self, length: np.ndarray, pos: np.ndarray, hi: np.ndarray) -> None:
        self.sess_len.append(length.astype(np.int32))
        self.sess_pos.append(pos.astype(np.int32))
        self.sess_hi.append(hi.astype(np.int32))

    def end_day(self, active: np.ndarray) -> None:
        self.daily_active.append(float(active.mean()))

    def reset_day(self) -> None:
        self.day_imp[:] = 0; self.day_score[:] = 0.0
        self.day_skip[:] = 0; self.day_novel[:] = 0

    def day_stats(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        imp = self.day_imp.astype(np.float64)
        den = np.maximum(imp, 1.0)
        return self.day_score / den, self.day_skip / den, self.day_novel / den, imp > 0

    def _engagement_by_len(self, length: np.ndarray, pos: np.ndarray) -> dict:
        engaged = pos > 0
        out = {}
        for (lo, hi), name in zip(BUCKETS, BUCKET_NAMES):
            m = (length >= lo) & (length <= hi)
            out[name] = float(engaged[m].mean()) if m.any() else float("nan")
        return out

    def finalize(self) -> dict:
        imp = np.maximum(self.total_imp, 1)
        length = np.concatenate(self.sess_len) if self.sess_len else np.array([0])
        pos = np.concatenate(self.sess_pos) if self.sess_pos else np.array([0])
        hi = np.concatenate(self.sess_hi) if self.sess_hi else np.array([0])
        p = self.cat_imp / max(self.cat_imp.sum(), 1)
        nz = p > 0
        repeat_mult = {f"repeat_click_mult_{k}": float(self.rep_click[k] / max(self.rep_imp[k], 1)
                                                       / (self.rep_click[0] / max(self.rep_imp[0], 1) or 1))
                       for k in (1, 2, 3)}
        return {
            "iwe_per_bot": self.iwe, "impressions": int(self.total_imp),
            "iwe_total": float(self.iwe.sum()), "iwe_mean": float(self.iwe.mean()),
            "print_summary": {
                "feed_click_rate": float(self.total_clicks / imp),
                "like_given_click": float(self.total_actions[1] / max(self.total_clicks, 1)),
                "follow_given_click": float(self.total_actions[3] / max(self.total_clicks, 1)),
                "forward_given_click": float(self.total_actions[2] / max(self.total_clicks, 1)),
                "longview_given_click": float(self.total_actions[0] / max(self.total_clicks, 1)),
                "hour_click_rate": {str(h): float(self.hour_click[h] / max(self.hour_imp[h], 1))
                                    for h in range(24)},
                **repeat_mult,
                "median_session_items": float(np.median(length)),
                "engagement_by_len": self._engagement_by_len(length, pos),
                "skip_rate": float((length - pos > 0).sum() / max(length.sum(), 1)),
            },
            "diagnostics": {
                "mean_session_depth": float(length.mean()),
                "high_intent_per_session": float(hi.mean()),
                "D1": self.daily_active[1] if len(self.daily_active) > 1 else float("nan"),
                "D3": self.daily_active[3] if len(self.daily_active) > 3 else float("nan"),
                "D7": self.daily_active[-1] if self.daily_active else float("nan"),
                "daily_active": list(self.daily_active),
            },
            "guardrails": {
                "cat_entropy_bits": float(-(p[nz] * np.log2(p[nz])).sum()),
                "catalog_coverage": float(self.seen_items.sum() / self.n_items),
                "cold_impression_share": float(self.cold_imp / imp),
                "cold_engagement": float(self.cold_eng / max(self.cold_imp, 1)),
                "repeat_exposure_rate": float(self.rep_total / imp),
            },
        }
