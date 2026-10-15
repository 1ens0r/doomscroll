"""Intuition: The bot psychology — a click is a sigmoid of taste-match, hidden quality, time of day
and fatigue; watch time is bimodal; high-intent actions are rare and conditional on the click;
leaving and returning are separate decisions (patience budget, next-day return).

All randomness is supplied as pre-drawn uniforms so every policy sees the same stream (CRN).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
ART = ROOT / "artifacts"
N_RESP = 7


def _sigmoid(x: np.ndarray | float) -> np.ndarray | float:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30.0, 30.0)))


def hour_effect_table() -> np.ndarray:
    """Real hourly click-rate log-odds, standardised to mean 0 / std 1 (shape only)."""
    with open(ART / "real_marginals.json") as f:
        table = json.load(f)["hour_click_rate"]
    rates = np.array([table[str(h)] for h in range(24)])
    logit = np.log(rates / (1.0 - rates))
    z = (logit - logit.mean()) / max(logit.std(), 1e-6)
    return z.astype(np.float32)


class ResponseModel:
    """Vectorised transition `click/day` for the active sessions of one lockstep step."""

    def __init__(self, cfg: dict) -> None:
        self.cfg = cfg["response"]
        self.hour = hour_effect_table()

    def p_click(self, match: np.ndarray, q: np.ndarray, hour: np.ndarray,
                sat_share: np.ndarray, prior_exp: np.ndarray, fatigue: np.ndarray) -> np.ndarray:
        c = self.cfg
        logit = (float(c["c0"]) + float(c["a_match"]) * (match - float(c["match_ref"]))
                 + float(c["b_quality"]) * (q - 0.5)
                 + float(c["hour_amp"]) * self.hour[hour]
                 - float(c["c_satiation"]) * sat_share
                 - float(c["repeat_penalty"]) * np.minimum(prior_exp, int(c["repeat_cap"]))
                 - float(c["d_fatigue"]) * fatigue)
        return _sigmoid(logit)

    def respond(self, match: np.ndarray, q: np.ndarray, hour: np.ndarray, sat_share: np.ndarray,
                prior_exp: np.ndarray, fatigue: np.ndarray, like_mult: np.ndarray,
                follow_mult: np.ndarray, forward_mult: np.ndarray, u: np.ndarray) -> dict:
        """u: (n, N_RESP) uniforms in [0,1); returns boolean/float outcome arrays."""
        c = self.cfg
        p_click = self.p_click(match, q, hour, sat_share, prior_exp, fatigue)
        click = u[:, 0] < p_click
        p_complete = _sigmoid(np.log(float(c["watch_p_complete"]) / (1 - float(c["watch_p_complete"])))
                              + float(c["watch_complete_slope"]) * (match - float(c["match_ref"]))
                              + float(c["watch_complete_slope"]) * (q - 0.5))
        complete = click & (u[:, 1] < p_complete)
        watch = np.where(complete,
                         float(c["watch_complete_lo"]) + u[:, 2] * (float(c["watch_complete_hi"])
                                                                    - float(c["watch_complete_lo"])),
                         float(c["watch_skip_lo"]) + u[:, 2] * (float(c["watch_skip_hi"])
                                                                - float(c["watch_skip_lo"])))
        watch = np.where(click, watch, 0.0)
        long_view = watch >= 0.7
        boost = 1.0 + (float(c["like_watch_boost"]) - 1.0) * long_view
        p_like = np.clip(float(c["p_like_base"]) * like_mult
                         * np.exp(float(c["p_like_slope"]) * (match - float(c["match_ref"]))) * boost,
                         0.0, 1.0)
        p_fwd = np.clip(float(c["p_forward_base"]) * forward_mult
                        * np.exp(float(c["p_highintent_slope"]) * (match - float(c["match_ref"])))
                        * (1.0 + 0.5 * long_view), 0.0, 1.0)
        p_fol = np.clip(float(c["p_follow_base"]) * follow_mult
                        * np.exp(float(c["p_highintent_slope"]) * (match - float(c["match_ref"])))
                        * (1.0 + 0.5 * long_view), 0.0, 1.0)
        like = click & (u[:, 3] < p_like)
        forward = click & (u[:, 4] < p_fwd)
        follow = click & (u[:, 5] < p_fol)
        p_hate = np.clip(float(c["hate_base"])
                         * np.exp(float(c["hate_mismatch_slope"]) * (1.0 - match)
                                  + float(c["hate_quality_slope"]) * (0.5 - q)), 0.0, 1.0)
        hate = click & (u[:, 6] < p_hate)
        tier = np.zeros(len(click), dtype=np.int8)
        tier[click] = 1
        tier[like] = 3
        tier[follow | forward] = 4
        score = (click.astype(np.float64) + 2.0 * long_view + 3.0 * like + 4.0 * forward
                 + 5.0 * follow)
        return {"click": click, "watch": watch.astype(np.float32), "long_view": long_view,
                "like": like, "follow": follow, "forward": forward, "hate": hate,
                "tier": tier, "pos": (score > 0).astype(np.float64), "score": score,
                "observed_play": click.astype(np.float64)}


def leave_step(budget: np.ndarray, tier: np.ndarray, is_repeat: np.ndarray, lcfg: dict) -> tuple:
    budget = budget - float(lcfg["step_cost"])
    budget = budget + np.where(tier >= 2, float(lcfg["reward_satisfying"]), 0.0)
    budget = budget - np.where(tier == 0, float(lcfg["penalty_skip"]), 0.0)
    budget = budget - np.where(is_repeat, float(lcfg["penalty_repeat"]), 0.0)
    return budget, budget <= 0.0


def return_prob(daily_sat: np.ndarray, new_frac: np.ndarray, skip_rate: np.ndarray,
                rcfg: dict) -> np.ndarray:
    logit = (float(rcfg["r0"])
             + float(rcfg["mean_sat_coef"]) * (daily_sat - float(rcfg["satisfaction_baseline"]))
             + float(rcfg["new_cat_coef"]) * new_frac
             - float(rcfg["regret_coef"]) * skip_rate)
    return _sigmoid(logit)
