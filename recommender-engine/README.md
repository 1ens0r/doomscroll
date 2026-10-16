# Recommender Engine

A from-scratch, end-to-end short-video recommender built on the **KuaiSAR-v2** dataset: offline
feature/model pipeline → realistic user-behaviour simulator → policy ladder → staged **Compose**
re-ranker. The point of the repository is *honesty*: every policy claim is backed by a paired
per-bot bootstrap confidence interval, a guardrail panel, and a reproducible negative result.

## Pipeline at a glance

```
prepare_data.py   KuaiSAR-v2 raw  ->  artifacts/ (features, item stats, user interest, scalers)   [Phase 0/1]
run_offline.py    train the offline TasteNet (models/taste.pt) + eval                             [Phase 1]
calibrate.py      bisect response/leave constants to real marginals -> artifacts/calibration.json  [Phase 2]
run_swarm.py      behaviour-sim policy ladder + parity gate -> artifacts/sim_results.json          [Phase 2]
run_learn.py      Generate->Train a sim-native ranker -> models/taste_sim.pt                        [Phase 3]
run_generations.py additive composer evolution -> artifacts/composer_generations.json               [Phase 3]
run_evolve.py     (1+lambda)-ES over policy knobs -> config/policy_evolved.yaml                     [Phase 4]
```

`feed/` is the simulator + serving library (`world`, `content`, `featurestate`, `response`,
`policies`, `composer`, `metrics`, `swarm`, `parity`, `learning`, `evolve`). Every tunable lives in
`config/` (`swarm.yaml`, `policy.yaml`, `policy_evolved.yaml`) — no magic numbers in code.

## Headline result — the policy ladder

10,000 bots × 7 days, seed 42, calibrated, paired per-bot bootstrap (2,000 resamples). IWE = the
platform's weighted-interaction metric (fixed weights, never tuned). Gap-closed is measured against
the personalized **oracle** upper bound.

| policy | IWE / bot | gap-closed | promotion |
|---|---|---|---|
| random | 61.86 | 0.00 | baseline |
| popularity | 77.87 | 0.11 | +16.0 |
| personal_affinity | 93.64 | 0.21 | +15.8 |
| taste (learned ranker) | 120.28 | 0.39 | +26.6 |
| taste_composer | 138.27 | 0.51 | +18.0 |
| **evolved (Phase 4)** | **149.54** | **0.58** | +11.3 |
| oracle (adaptive upper bound) | 213.10 | 1.00 | +74.8 |

Every adjacent rung's 95% CI excludes zero → **ladder PASS**. The `Compose` stage adds a further
+13.4 IWE/bot over `taste`, and the Phase-4 **(1+λ) evolution** over composer + score weights +
retrieval mix adds **+11.3** more, while *improving* every guardrail (catalog coverage 75→82%,
category entropy 3.35→3.65 bits, repeat-exposure 0.57→0.52%). Details in `../notes/phase2.md`,
`../notes/phase3.md` and `../notes/phase4.md`; the visual version is
[`../notes/figures/01_ladder.png`](../notes/figures/01_ladder.png), rendered by `feed/plotting.py`.

## What makes it credible

- **Feature parity gate** — 13/13 replayable user-side features reproduced within 15% by an
  independent `_replay_user` implementation against real `features_test` rows (`feed/parity.py`).
- **Calibration** — response/leave constants fit to real marginals (7/12 targets; the 5 misses are
  documented structural model-form limits, not hidden).
- **No leakage** — retrieval and features use observable history only; the learned ranker trains on
  sim logs with a by-bot split and train-only scalers; no oracle distillation.
- **Guardrails + negative results** — the composer evolution keeps only CI-separated gains and
  records the rejected components (seen-penalty, cold bonus, epsilon-greedy) with their measured cost;
  the Phase-4 (1+λ)-ES discounts IWE by 5% per violated guardrail, and the winning knobs
  (`config/policy_evolved.yaml`) improve coverage and diversity rather than trading them away.

## Run

```bash
cd recommender-engine
P=~/miniconda3/envs/picograd/bin/python
$P -m pytest tests/ -q            # fast unit + parity tests
$P run_swarm.py --fast            # smoke ladder (~25 s)
$P run_swarm.py                   # full ladder incl. evolved (~130 s) -> artifacts/sim_results.json
$P run_generations.py             # composer generations (~209 s)
$P run_evolve.py --fast           # smoke evolve loop (~10 s)
$P run_evolve.py                  # full (1+lambda)-ES (~335 s) -> config/policy_evolved.yaml
```

See `ASSUMPTIONS.md` for every modelling choice and `../notes/` for per-phase write-ups.
