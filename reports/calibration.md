# Phase 2 — Calibration report

`calibrate.py` fits the response/leave constants by coordinate-descent bisection against the real
marginals in `config/swarm.yaml` (fitted with the `popularity` policy, 2,000 bots, 3 days, seed 42,
3 passes, 225 s). Fitted constants are written to `artifacts/calibration.json`; `world.load_config()`
layers them over `swarm.yaml`. Targets are never edited by code.

**Result: 7 / 12 targets within 15%** (`pass_rate = 0.583`). The misses are structural, not tuning
failures — see below. (This is a genuine drop from the pre-world-fix 9/12: the user-conditional
appeal changes the engagement distribution, and the fitter now refuses to chase five mutually
inconsistent targets.)

## Sim vs real (calibrated, popularity policy)

| target | sim | real / target | rel err | verdict |
|---|---|---|---|---|
| feed click rate | 0.4938 | 0.538 | 8.2% | PASS |
| like \| click | 0.0423 | 0.040 | 5.8% | PASS |
| follow \| click | 0.00547 | 0.0055 | 0.6% | PASS |
| forward \| click | 0.00449 | 0.0042 | 6.9% | PASS |
| hour peak (click) | 0.585 | 0.55 | 6.4% | PASS |
| hour trough (click) | 0.327 | 0.23 | 42.0% | **FAIL** |
| repeat ×1 | 0.270 | 0.293 | 7.8% | PASS |
| repeat ×2 | 0.0588 | 0.203 | 71.0% | **FAIL** |
| repeat ×3 | 0.000 | 0.079 | 100% | **FAIL** |
| median session items | 7 | 8 | 12.5% | PASS |
| engagement, 1-item sessions | 0.143 | 0.191 | 25.2% | **FAIL** |
| engagement, 31–100 item sessions | 1.000 | 0.50 | 100% | **FAIL** |

Fitted constants: `c0=-0.620`, `p_like_base=0.01973`, `p_follow_base=0.00327`,
`p_forward_base=0.00211`, `hour_amp=0.30`, `repeat_penalty=2.254`, `patience_median=8.3125`,
`reward_satisfying≈0`, `penalty_skip=0.640`.

## Why the five misses are structural

1. **Hour trough (0.327 vs 0.23).** The README's hourly *engagement* shape has a deep 03–04h crater;
   our only hourly source in the artifacts is the hourly *click* rate, which is much flatter
   (min 0.494 vs mean 0.532). A z-scored click-logit effect cannot produce a deep trough without
   also flattening the peak. The shape, not the amplitude, is the limit; `hour_amp` fits the peak.
2. **Repeat ×2 / ×3 (0.059 / 0 vs 0.203 / 0.079).** The README's repeat multipliers decay
   sub-linearly across *days*, while the model applies a single per-session `repeat_penalty`; the
   simulator's session-level satiation suppresses near-term re-exposure far more than real multi-day
   feeds do. ×1 fits (7.8%); ×2 and ×3 cannot without a cross-day re-exposure process the model lacks.
3. **Engagement, 1-item sessions (0.143 vs 0.191).** The sim leave model is geometric, so it
   under-produces the real point-mass of one-item sessions.
4. **Engagement, 31–100 items (=1.0 vs 0.50).** Session length *is* driven by engagement (skips cost
   budget, satisfying items don't), so by construction only highly-engaged sessions survive past
   ~30 items. The real curve rises 0.19→0.50 and saturates; reproducing it needs length variation
   partly independent of engagement, which the leave model does not have. Documented rather than gamed.

The seven matched targets anchor click rate, conditional-action rates, the repeat ×1 rate and the
median session, so the swarm speaks in real units for the quantities that matter for the policy
comparison.

## Guardrails under the calibrated popularity policy

`cat_entropy = 3.76 bits` / `log2(38) = 5.25`; catalog coverage 1.1% (2,000 bots / 3 days);
repeat-exposure rate 1.72%; cold impression share 0% at this horizon. Guardrail levels are reported,
not gated, because the plan sets no numeric thresholds for them.

## Reproduce

```bash
cd recommender-engine
~/miniconda3/envs/picograd/bin/python calibrate.py        # ~225 s, writes artifacts/calibration.json
~/miniconda3/envs/picograd/bin/python run_swarm.py --fast # smoke; full run ~93 s
```
