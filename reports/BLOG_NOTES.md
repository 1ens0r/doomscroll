# BLOG NOTES — 10 story beats

Ten beats for the write-up. Each carries its own real numbers straight from the committed
artifacts; the honest-claims rules (§ "Honesty beats") apply to every beat. Companion material:
`reports/phase1–4.md`, `reports/calibration.md`, `reports/GENERATIONS.md`,
`reports/figures/*.png`, `status.md`.

---

## Beat 1 — Intent is the rare thing: only ~5% of the feed carries it

Tier 0 (pure scroll, no engagement) is **51.82% · 6.77M events**; every number is a decision waiting
to happen. Clicks are cheap (**43.24%**), high intent is gold: like **2.34%**, follow/forward
**0.59%**. So we rank on a graded `label_tier` (weights: click 1 → follow 5, hate −3, plus a
+2.0 search-click bonus, mean tier-4 score 6.42) instead of a 95%-noise + 5%-gold binary label.
The funnel — feed click **53.79%**, conditional like **~4.0%**, follow **~0.55%** — is
three different intents, not one scale.

*Figure: `eda/figures/02_tier_distribution.png`, `eda/figures/03_action_cooccurrence.png`.*

## Beat 2 — No lone feature wins; habit + affinity does

Top single features: any-engagement AUC `u_pos_rate` **0.683**; strong-engagement `u_rate_like`
**0.832** — but they are the same 5% of users! Group ranking: **user > user-category > item >
context > user-item > cold**. Verdict: a 43-feature set (habit `u_*`, affinity `uc_*`, quality
`i_*`, memory `ui_*`, context `ctx_*`), with response curves that are cleanly monotone for
`uc_decay_share`, `uc_lift`, `u_pos_rate` — ideal for GBDT-shaped learners.

*Figure: `eda/figures/03_feature_viability.png`.*

## Beat 3 — Activity is brutalist inequality → personalize or die

Median 223 events/user, Gini **0.63**: the top **1%** of users generate **12.0%** of all events,
the top 10% take **47.9%**. Global estimators collapse here; every model input in the engine is
user-normalized, and the simulator stratifies real users into activity terciles.

*Figure: `eda/figures/03_user_activity.png`; cohort payoff in `reports/figures/04_cohorts.png`.*

## Beat 4 — Exposure ≠ quality: popularity is a coverage hack

Rank-frequency is a clean log-log long tail; the top **1%** of items grab **20.3%** of impressions,
but log-exposure vs engagement-rate correlates at only **0.128**. The ladder confirms it in
simulation: `popularity` buys only **+16.0 IWE** (+10.6% of the oracle gap) over random, and its
catalog coverage collapses to **5.9%** with entropy 4.08 bits.

*Figure: `eda/figures/03_item_popularity.png`; ladder in `reports/figures/01_ladder.png`.*

## Beat 5 — Search is the cheat code for feed ranking

Same-category search history lifts feed engagement **+6–10pp** (49.3% → 55.6% with search only;
54.7% → 64.3% with both priors). Search and feed run on opposite clocks (search peaks 03–05h,
feed 20–21h) — different context, same user. Hence search-derived features + a +2.0 tier bonus,
and search click-logs being click-only handled via `schema.json: observed_by_source` — zeros mean
*unlogged*, not negative.

*Figure: `eda/figures/03_search_vs_feed.png`.*

## Beat 6 — Validate policies on simulated users, side effects included

Before shipping a policy through real estimation, the engine replays it through a
**10,000-bot × 7-day** lockstep simulator with **common random numbers**: same bots, same
sessions, same draws. Policies scored this way are paired experiments — the ladder is
`random 61.86 → popularity 77.87 → personal_affinity 93.64 → taste 120.28 → taste_composer 138.27
→ evolved 149.54 → oracle 213.10`, and **every adjacent rung's 95% CI excludes zero**.

*Figure: `reports/figures/01_ladder.png`; simulator: `feed/swarm.py`.*
*Honesty: the sim is calibrated to marginals only, so absolute IWE is meaningless — relative
claims under stated assumptions is the rule.*

## Beat 7 — The parity gate: the sim must speak in real units

A separate `_replay_user` implementation recomputes the **13 replayable user-side features** from
interactive history and must match real `features_test` rows within **15%** — 13/13 PASS, from
`u_pos_rate` (5.8% err) down to `ctx_hour_sin` (**5.2e-9** — the deterministic features match to
floating-point). Five known-partial features are reported with reasons (history warm-up, item-side
excluded), not hidden. The gate lives in `feed/parity.py` and runs on the ladder.

*Raw: `artifacts/sim_results.json:parity`.*

## Beat 8 — Calibrate to marginals — and air the misses

`calibrate.py` bisects response/leave constants onto the real marginals: **7/12 targets inside
15%** (click rate 8.2%, like|click 5.8%, follow|click 0.6%, forward|click 6.9%, hour peak 6.4%,
repeat ×1 7.8%, median session 12.5%). The **5 misses are structural, not tuning failures**: the
hour *shape* the click-logit can't produce, the cross-day repeat ×2/×3 process the session model
lacks, and the engagement-vs-length curve that is tautological in a leave model driven by
engagement. Documented in `reports/calibration.md` — the fitter refuses to chase inconsistent
targets, and passes 7.

*Figure: `reports/figures/03_calibration.png`.*

## Beat 9 — Compose: cheap numpy beats a clever deep stage

The **Compose** stage is a fast re-rank applied on every swipe, decoed from the ranker:
penalize in-session duplicates (hard ban), punish category runs (α), reward what the current
session is actually up-rating (η). Staged additively, gated by paired bootstrap AND no guardrail
degrading >10%: **KEEP α +4.45 [+2.91, +5.96] and η +8.98 [+7.61, +10.39]**; **REJECT** the seen
penalty (−0.01, redundant), the cold bonus (−1.04) and ε-greedy (−2.34, which also trips the
repeat guardrail +21%). `taste_composer` = **138.27**, 0.51 of the gap, dominating `taste` on
every guardrail.

*Figure: `reports/figures/02_generations.png`; log: `reports/GENERATIONS.md`.*

## Beat 10 — Evolution with a TEST discipline: +11.3 IWE, all guardrails better

A `(1+λ)` ES over **16 knobs** (composer ×7, score weights ×5, retrieval mix ×4) with
`fitness = IWE − 5%·IWE per guardrail violated >10%`. Search on TRAIN (2k×3d, seed 1, monotone
58.08 → 62.11), select on VAL (5k×7d, seed 2, best 149.84), **TEST exactly once** (10k×7d, seed 3:
**150.03 vs 136.50, +13.53 [+12.10, +14.92]`). Evolution independently rediscovers Phase-3's
direction (α 0.4 → 0.92, η 0.5 → 1.16) and confirms the three dead knobs stay ≈ 0. On the
seed-42 ladder `evolved` = **149.54** (gap-closed **0.58**), with coverage 75 → 82%, entropy
3.35 → 3.65 bits, repeat exposure *down* 0.57 → 0.52%. Guardrails got better because the loss
surface rewarded personalization, not exposure gaming.

*Figure: `reports/figures/01_ladder.png`, `reports/figures/04_cohorts.png`,
`reports/figures/05_retention.png`; method: `reports/phase4.md`.*

---

## Honesty beats (append as-is)

- The simulator is hand-designed and calibrated to marginals only (7/12); **absolute IWE is not a
  claim about real users** — results are relative, under stated assumptions (A0–A21 in
  `ASSUMPTIONS.md`).
- The world was deliberately built so preference-alignment pays (A18); popularity is a weak rung
  by construction, the oracle's 6.5% repeat-exposure rate is a greedy artifact, not a target.
- Phase-4 fitness and IWE coincided (penalty never fired) — the penalty is a gate that stayed
  healthy rather than a discriminator.
- `personal_affinity` is honest on light users (A0 gap 21%), but its cohort tail shows diversity
  pressure (A0_B2 **−15%** in `04_cohorts.png`): personalization has a floor term, and the
  composer claws it back.

> One sentence: *we model interaction intent, session dynamics and retention drivers on
> KuaiSAR-v2 — then rank in a calibrated behavior-sim where every rung must clear a paired
> bootstrap, a parity gate and a guardrail panel, and where "it didn't work" is data.*
