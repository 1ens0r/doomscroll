# Phase 2 — World + swarm

**Verdict: PARTIAL.** The simulator, feature-parity check and calibrated psychology all work;
the feature-parity gate passes 13/13 replayable user-side features, calibration passes 9/12 targets,
and the IWE ladder passes **2 of 4** adjacent rungs at full scale. The two failing rungs have a
single diagnosed cause: in this calibrated world, engagement is dominated by *global item appeal*,
so pure popularity is already ~98% of the oracle and neither heuristic nor learned personalization
beats it. Details and evidence below.

## What was built

| file | role |
|---|---|
| `feed/world.py` | `Catalog` (50k items, hidden appeal `q`, live visible stats, daily 5% replacement), `Bots` (hidden taste from `user_interest`, activity tiers, per-wave session schedule, patience/return CRN), `World.rng(*key)` position-independent streams |
| `feed/featurestate.py` | live decayed bot/`uc`/item counters + `ui` seen memory; `pair_features` emits the exact 27 dense + `cat_id` features the offline model consumes; `_replay_user` pure-python parity replay |
| `feed/response.py` | bot psychology: `p_click` (taste-match + hidden quality + hour + satiation + repeat + fatigue), bimodal watch, rare high-intent actions, `leave_step`, `return_prob` |
| `feed/policies.py` | `TasteScorer` (TasteNet inference), `build_pool` (Find), `rank_scores` (random / popularity / personal_affinity / taste / oracle) |
| `feed/composer.py` | fast "Compose" re-ranker (novelty, category runs, session affinity); used by `taste_composer` and the adaptive oracle |
| `feed/metrics.py` | IWE, session depth, hourly/repeat curves, engagement-by-length, D1/D3/D7, diversity/coverage/cold guardrails |
| `feed/swarm.py` | lockstep wave engine (Find→Rank→Compose→response→feedback) with shared random numbers |
| `feed/parity.py` | replays sampled real users through an independent feature implementation and compares to `features_test` |
| `calibrate.py` | coordinate-descent bisection fit of response/leave constants → `artifacts/calibration.json` |
| `run_swarm.py` | ladder run: paired per-bot bootstrap CIs, oracle-gap-closed, cohorts, guardrails, acceptance gate → `artifacts/sim_results.json` |
| `tests/test_phase2.py` | 7 fast tests: world determinism, policy-independent schedule, response bounds/tiers, leave/return bounds, IWE math, score shapes, parity gate |

## IWE ladder (full scale: 10,000 bots × 7 days, seed 42, calibrated)

Ranking score is `1·click + 2·watch + 3·like + 4·forward + 5·follow`; IWE weights are fixed
(click 1, long-view 2, like 3, forward 4, follow 5, hate −3) and never tuned.

| policy | IWE / bot | 95% paired CI vs previous | gap-closed | verdict |
|---|---|---|---|---|
| random | 62.02 | — | 0.00 | baseline |
| popularity | 86.33 | +24.32 [+23.65, +25.03] | 0.98 | PASS |
| personal_affinity | 64.76 | −21.57 [−22.30, −20.88] | 0.11 | **FAIL** |
| taste | 64.95 | +0.19 [−0.31, +0.71] | 0.12 | **FAIL** (CI spans 0) |
| oracle | 86.84 | +21.89 [+21.17, +22.63] | 1.00 | PASS |

| adjacent rung | verdict |
|---|---|
| random < popularity | PASS |
| popularity < personal_affinity | FAIL |
| personal_affinity < taste | FAIL (positive but CI spans 0) |
| taste < oracle | PASS |

Full `iwe`/`gap`/`ladder`/`cohorts` JSON is in `artifacts/sim_results.json`.

## Feature parity (live serving path vs offline features)

`parity.py` replays 150 sampled real users through the independent `_replay_user` implementation and
compares to their `features_test` rows (6,007 matched rows).

- **13 / 13 replayable user-side features within 15%** (gate PASS): `u_pos_rate`, `u_mean_score`,
  `u_rate_click/like`, `uc_decay_share`, `uc_lift`, `uc_exposure_share`, `uc_score`, `uc_recency`,
  `u_recency`, `ctx_hour_sin/cos`, `ui_prior_max_tier`. `ctx_hour_*` matches to 0.0 after fixing a
  missing 8h timezone offset used by `build.py`.
- **8 counters reported as partial (not gated), with reasons** — `u_n_events`, `uc_repeat`,
  `ui_prior_n` (computed on the full timeline *before* the time split, so the pre-split 50% warmup
  is not stored in any split file), `cold_user/uc` (rareness follows from the same missing warmup),
  `u_rate_follow/forward` (rare-event means sensitive to that warmup), `u_play_ratio` (artifacts
  store `play_ratio` clipped to `watch` in [0,1]).
- Item-side `i_*` features are excluded by design: a user sample cannot reproduce the global item
  impression stream.

## Cohorts and guardrails (popularity vs taste)

| cohort (activity tercile × taste-breadth tercile) | popularity | taste |
|---|---|---|
| light users (A0) | 26.1 | 20.8 |
| heavy users (A2) | 162.4 | 120.9 |
| A0_B0 → A2_B2 | 24.4 → 168.2 | 19.8 → 125.1 |

Guardrails (popularity, full run): category entropy 4.34 bits (`log2 38 = 5.25`), catalog coverage
7.7%, cold-impression share 0.0%, repeat-exposure rate 0.78%.

## Findings — why the ladder is partial

1. **Popularity ≈ oracle in this world (86.33 vs 86.84, gap-closed 0.98).** The response model rewards
   hidden item appeal `q`, and the `popularity` ranker scores items by their smoothed *observed*
   quality (`item_popularity ≈ mean intent score`), which is initialized directly from the real
   per-item aggregates and therefore tracks `q` almost perfectly. There is almost no headroom left
   for personalization to win.
2. **The Phase-1 model does not transfer to the sim.** Offline the TasteNet ranks well
   (`corr(score, tier)=0.39`, `corr(score, i_mean_score)=0.40`, `corr(score, uc_lift)=0.13`), but
   in-sim `corr(score, true category match) ≈ 0.0`, so the `taste` rung is effectively a noisy
   quality ranker (IWE 64.95). Root cause: in-sim bot history counters start at zero while the
   offline features were computed over full history (`u_n_events` mean ≈ 2120 offline vs tens
   in-sim), so the model is out of distribution.
3. **Attempted fix (time-boxed, rejected).** Seeding each bot's `FeatureState` from TRAIN-window
   per-user aggregates (no test leakage) moved quality ranking (in-sim `corr(score, q)` 0.03→0.24)
   but did **not** recover personalization: `corr(score, match)` stayed ≈ −0.01, and live-feature
   means were still ~50% of the test means because train-window users are much less mature than the
   test window (temporal drift). Pass check (`corr ≥ 0.3` and parity within 15%) failed, so per the
   agreed fallback the verbose seeding code was not merged; see `ASSUMPTIONS.md` A16.
4. **Personalization signal is genuinely weak here.** Even the raw-ish signal available to rankers,
   `uc_lift`, correlates only ~0.17 with true category match in-sim, and users' preferred categories
   (from real `decay_share`) are not the high-`q` categories, so personalizing *lowers* mean item
   quality faster than it raises match.

**Implication (honest):** this simulator supports "popularity is strong and hard to beat", which is a
real and well-known phenomenon, but it cannot currently demonstrate "personalization beats
popularity". Making that claim would require either a user-conditional item-appeal model (so `q`
depends on the user) or retraining the Taste model on simulator-generated logs — both are follow-up
work, not Phase-2 scope.

## Reproduce

```bash
cd recommender-engine
~/miniconda3/envs/picograd/bin/python -m pytest tests/test_phase2.py -q   # 7 tests, ~3 s
~/miniconda3/envs/picograd/bin/python run_swarm.py --fast                 # smoke (~20 s)
~/miniconda3/envs/picograd/bin/python run_swarm.py                        # full (~95 s) → artifacts/sim_results.json
```
