# Phase 3 — Composer generations

**Verdict: PASS.** Additive composer generation over a fixed neutral baseline, gated on a paired
per-bot bootstrap IWE gain (CI excludes 0) **and** no tracked guardrail degrading more than 10%.
Two of five components earned their keep; three were honestly rejected. The tuned composer lifts the
policy from `taste` = 120.28 to **`taste_composer` = 138.27** IWE/bot (gap-closed 0.39 → 0.51),
+13.44 (+11.2%) over the neutral composer, all while improving every guardrail relative to `taste`.

Full per-component table: `notes/GENERATIONS.md`; raw JSON: `artifacts/composer_generations.json`.

## Method — additive generations

`run_generations.py` starts from a **neutral** composer (no seen-penalty, no diversity, no session
adaptation, no cold bonus, no exploration) and turns on **one** knob at a time, keeping it only if it
passes the gate. This is a drop-one-in sequence, not a grid search: each accepted component becomes
the new baseline for the next.

| gate | rule |
|---|---|
| benefit | paired bootstrap 95% CI on ΔIWE/bot excludes 0 and Δ > 0 |
| safety | `catalog_coverage` and `repeat_exposure_rate` may not degrade > 10% vs the accepted baseline |

Rejected components stay in the code path (knob = 0) and are recorded with their measured Δ, so the
negative results are reproducible, not deleted.

## Results (10,000 bots × 7 days, seed 42, calibrated, 2,000 bootstraps)

| gen | component | knob | ΔIWE vs accepted | 95% CI | guardrail | verdict |
|---|---|---|---|---|---|---|
| 1 | seen-penalty | `gamma=2.0` | −0.01 | [−0.66, +0.64] | ok | **REJECT** (no IWE gain) |
| 2 | diversity | `alpha=0.4` | +4.45 | [+2.91, +5.96] | ok | **KEEP** |
| 3 | session adaptation | `eta=0.5` | +8.98 | [+7.61, +10.39] | ok | **KEEP** |
| 4 | cold bonus | `beta=0.3` | −1.04 | [−2.21, +0.13] | ok | **REJECT** (no IWE gain) |
| 5 | epsilon exploration | `epsilon=0.05` | −2.34 | [−3.00, −1.67] | bad | **REJECT** (no IWE gain) |

Accepted set → `config/policy.yaml`:
`gamma=0.0, alpha=0.4, eta=0.5, beta=0.0, epsilon=0.0, decay=0.8, window_m=5`.

## Findings

1. **Diversity and session-adaptation are the whole story (+13.4).** `alpha` (category-run penalty)
   and `eta` (in-session taste affinity) are the only components with a clear, CI-separated gain.
   Session adaptation is the larger of the two: within a session, re-weighting toward the user's
   revealed preference pays off more than spreading the feed.
2. **The seen-penalty is redundant (−0.01).** `ComposerState.select` already hard-bans items shown
   earlier in the *same* session, and across-session re-exposure is already rare (repeat-exposure
   rate 0.57%). Adding a soft cross-session penalty buys nothing measurable — a clean negative
   result that validates the existing within-session dedup.
3. **The cold bonus does not improve IWE (−1.04).** Boosting cold items mildly *costs* engagement in
   this world. Note this is an IWE decision only: cold-impression share is a separate P0 concern and
   the coverage guardrail stays healthy (78.5% catalog coverage with `beta=0`), so the cold-start
   machinery is retained and exposed as a knob rather than deleted.
4. **Random exploration hurts (−2.34) and trips a guardrail.** Epsilon-greedy exploration lowers IWE
   and degrades the repeat-exposure guardrail — there is no exploration benefit to justify the cost
   at this scale.
5. **Guardrails improve, not just hold.** Relative to `taste`, `taste_composer` raises category
   entropy (3.35→3.47 bits), catalog coverage (75.2→78.5%), cold-engagement (0.68→0.71) and lowers
   repeat-exposure (0.57→0.54%). The escaper never trades a guardrail for IWE.

## Reproduce

```bash
cd recommender-engine
~/miniconda3/envs/picograd/bin/python -m pytest tests/test_phase3.py -q   # fast
~/miniconda3/envs/picograd/bin/python run_generations.py --fast            # smoke (~23 s)
~/miniconda3/envs/picograd/bin/python run_generations.py                   # full (~209 s) → artifacts/composer_generations.json
```
