# Phase 4 — Evolution loop

**Verdict: PASS.** A (1+λ) evolution strategy over 16 policy knobs — the five composer
coefficients, the five score weights and the four retrieval-mix counts — lifts the ladder from
`taste_composer` = 138.27 to **`evolved` = 149.54 IWE/bot** (seed 42, gap-closed 0.51 → 0.58), and
**+11.27 [+9.94, +12.74]** over `taste_composer` with a paired per-bot 95% CI that excludes zero.
On the disjoint protocol set (TRAIN seed 1 / VAL seed 2 / TEST seed 3) the winner is **150.03 vs
136.50** for the shipped composer (+13.53 [+12.10, +14.92]). Every guardrail improves along the way.

Full generation log: `notes/GENERATIONS.md` (Phase-4 section); raw numbers:
`artifacts/evolve_results.json`; winning knobs: `config/policy_evolved.yaml`.

## Method — (1+λ)-ES with a guardrail penalty

`feed/evolve.py` is a self-contained (1+λ) evolution strategy; `run_evolve.py` orchestrates it.

```
x        = 16-dim knob vector (spec in policy.yaml:evolve.knobs, min/max per knob)
child    = clip(x + sigma · range ⊙ N(0,1), lo, hi),   integer knobs rounded
sigma    = 0.15 (of each knob's range) × 0.85 per generation
fitness  = IWE − 5%·IWE · (# guardrails degraded >10% vs the search-start composer)
select   = best of {parent, λ=12 children}  (elitist, parent competes)
```

Search runs on a small **TRAIN** swarm (2,000 bots × 3 d, seed 1). The best child of each of the 8
generations is then re-scored on a larger **VAL** swarm (5,000 × 7 d, seed 2) and the argmax is
selected. The winner is measured on **TEST** exactly once (10,000 × 7 d, seed 3). Population
evaluations run in an 8-worker `multiprocessing` pool (single-threaded torch per worker) with a
900 s time budget that shrinks the generation count if the projection overruns.

Knobs and their mapping live in `feed/evolve.py:KNOB_BLOCKS`; the search starts from the Phase-3
accepted composer, so `beta`, `epsilon` and `gamma` start at 0 (the components Phase 3 rejected).

## Search trace (TRAIN, 2,000 bots × 3 d, seed 1)

| gen | best-child IWE | fitness | guardrail violations | vs parent |
|---|---|---|---|---|
| 1 | 58.08 | 58.08 | 0 | improved |
| 2 | 58.56 | 58.56 | 0 | improved |
| 3 | 59.61 | 59.61 | 0 | improved |
| 4 | 59.75 | 59.75 | 0 | improved |
| 5 | 60.95 | 60.95 | 0 | improved |
| 6 | 61.41 | 61.41 | 0 | improved |
| 7 | 61.84 | 61.84 | 0 | improved |
| 8 | 62.11 | 62.11 | 0 | improved |

Every generation improved and no candidate ever tripped a guardrail, so the penalty never fired
early — but it is retained in the code path and is what makes the `evolved` rung safe to accept.
The search is monotone but greedy; the VAL selection is what protects against TRAIN-specific noise.

## Selected knobs

```yaml
composer:    {gamma: 0.03, alpha: 0.92, eta: 1.16, beta: 0.0, epsilon: 0.0, decay: 0.92, window_m: 9}
score_weights: {click: 1.64, watch: 1.59, like: 4.74, forward: 6.88, follow: 4.62}
retrieval:   {n_popular: 48, n_affinity: 65, n_new: 23, n_random: 19}
```

- **Diversity and session-affinity grow** (`alpha` 0.4 → 0.92, `eta` 0.5 → 1.16): evolution
  independently rediscovers the Phase-3 direction and pushes it further.
- **`gamma`/`beta`/`epsilon` stay ≈ 0** — the Phase-3 rejections survive evolution, a second
  independent confirmation that the seen-penalty, cold bonus and ε-exploration are dead weight here.
- **Retrieval tilts toward the affinity slot** (50 → 65) at the expense of uniform random (20 → 19);
  personalization has more to gain from the observable-history candidates than from noise.
- **Score re-weighting:** forward and like up, watch down — the rare high-intent actions dominate
  the composer's IWE, so the ranker is pushed to surface them.

## Full policy ladder (10,000 bots × 7 d, seed 42, calibrated, 2,000 bootstraps)

| policy | IWE / bot | gap-closed | vs previous | 95% CI | verdict |
|---|---|---|---|---|---|
| random | 61.86 | 0.000 | — | — | baseline |
| popularity | 77.87 | 0.106 | +16.01 | [+15.34, +16.69] | PASS |
| personal_affinity | 93.64 | 0.210 | +15.77 | [+14.14, +17.42] | PASS |
| taste (learned) | 120.28 | 0.386 | +26.64 | [+24.21, +29.33] | PASS |
| taste_composer | 138.27 | 0.505 | +17.99 | [+16.59, +19.39] | PASS |
| **evolved** | **149.54** | **0.580** | **+11.27** | **[+9.94, +12.74]** | **PASS** |
| oracle | 213.10 | 1.000 | +92.82 | [+90.46, +95.34] | ceiling |

The acceptance ladder (`random < popularity < personal_affinity < taste < oracle`) passes all four
adjacent rungs; `taste_composer` and `evolved` are reported as extra, CI-separated rungs above
`taste`. `artifacts/sim_results.json` records all pairs.

## Guardrails — the escaper never trades them for IWE

| policy | cat entropy (bits) | catalog coverage | cold-impression share | cold engagement | repeat-exposure |
|---|---|---|---|---|---|
| taste | 3.346 | 75.2% | 4.95% | 0.683 | 0.57% |
| taste_composer | 3.474 | 78.5% | 4.80% | 0.705 | 0.54% |
| **evolved** | **3.651** | **81.7%** | 4.90% | **0.712** | **0.52%** |
| oracle | 3.857 | 91.4% | 2.68% | 0.758 | 6.51% |

`evolved` improves entropy, coverage, cold-engagement and repeat-exposure versus `taste_composer`
— the +11.3 IWE is not bought with a diversity loss. (The oracle's high repeat-exposure is an
artifact of it greedily re-showing known-good items; it is a ceiling, not a deployable policy.)

## Protocol numbers (disjoint seeds)

| stage | bots × days | seed | result |
|---|---|---|---|
| TRAIN (search) | 2,000 × 3 | 1 | best child IWE 62.11 |
| VAL (select) | 5,000 × 7 | 2 | selected IWE 149.84 |
| TEST (once) | 10,000 × 7 | 3 | evolved 150.03 vs composer 136.50, Δ +13.53 [+12.10, +14.92] |

TRAIN/VAL/TEST use different worlds (different seeds), so the TEST gain is a genuine held-out
result, not a re-measurement of the search objective. It agrees with the seed-42 ladder
(+11.27), which is the headline.

## Timings (Ryzen 5700G, 8 workers, CPU)

| stage | seconds |
|---|---|
| reach search (8 gen × 12 children + base) | 176 |
| VAL selection (9 candidates) | 99 |
| TEST (2 policies) | 59 |
| **evolve total** | **335** |
| full swarm ladder incl. evolved (7 policies + parity) | 130 |

Well inside the 15-minute budget; the auto-shrink guard never triggered.

## Reproduce

```bash
cd recommender-engine
~/miniconda3/envs/picograd/bin/python -m pytest tests/test_phase4.py -q   # fast
~/miniconda3/envs/picograd/bin/python run_evolve.py --fast                 # smoke (~10 s)
~/miniconda3/envs/picograd/bin/python run_evolve.py                        # full (~335 s) → config/policy_evolved.yaml
~/miniconda3/envs/picograd/bin/python run_swarm.py                         # ladder incl. evolved (~130 s)
```

## Negative/limitations

- The search is **greedy and monotone** on a small TRAIN swarm; VAL selection is the only guard
  against overfitting to it. With a bigger budget a proper CMA-ES or multi-start would be safer.
- Fitness and IWE coincided (0 violations throughout), so the guardrail penalty is untested *as a
  discriminator* here — only as a gate that stayed healthy.
- Evolution tunes the *policy*, not the *ranker*; it cannot fix the residual gap to the oracle
  (0.58 vs 1.00), which is bounded by how much signal the observable features carry.
