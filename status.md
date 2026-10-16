# Build status — phase by phase

A from-scratch, CPU-only short-video recommender on **KuaiSAR-v2**: raw data → offline model →
calibrated behaviour simulator → policy ladder → staged **Compose** re-ranker → **evolution loop**.
Every policy claim is backed by a paired per-bot bootstrap CI, a guardrail panel, and a reproducible
negative result. This file tracks what is **built** vs **left**.

_Last updated: Phase 6 + Phase A (revival plan). Phases 0–5 committed; Phase 6 audit and the Phase-A
improvements (bigger network, ranking-aware loss, composer refresh, executed `demo.ipynb`) added._

## The whole project in ten lines

1. KuaiSAR-v2 19.62M clean events → 13.07M k-core rows; 43 leakage-free as-of features, strictly temporal splits.
2. Only ~5% of events carry high intent → rank on a graded `label_tier`, never on binary clicks.
3. Activity Gini 0.63 (top 1% = 12% of events) → every input user-normalized; personalization is the thesis.
4. Phase 1: TasteNet (32k params), graded NDCG@10 0.496 val / 0.519 test, beats every hand baseline (LightGBM ceiling bounded).
5. Phase 2: 10k-bot lockstep behavior sim on shared random numbers; calibration 7/12 with the 5 misses documented; parity gate 13/13.
6. Phase 3: Compose kept α-diversity (+4.45) + η-session-affinity (+8.98); REJECTED γ / β / ε with measured costs → 138.27 IWE.
7. Phase 4: (1+λ)-ES over 16 knobs, guardrail-penalised, TRAIN→VAL→TEST-once: +13.53 [+12.10, +14.92] on TEST.
8. Ladder: random 61.86 → evolved 149.54 → oracle 213.10 IWE/bot; 0.58 of the personalized gap closed, every adjacent CI separated.
9. IWE rose because personalization paid, not exposure gaming: coverage 75→82%, cat-entropy 3.35→3.65 bits, repeat 0.57→0.52%.
10. Claims stay relative by construction: calibrated hand-designed sim, paired per-bot bootstraps, negative results kept with numbers.

## At a glance

| Phase | Deliverable | State | Headline evidence |
|---|---|---|---|
| 0 | Data prep: raw → `artifacts/` | ✅ DONE (`2023ba8`) | 19.6M events → 13.06M k-core rows; `artifacts/*.parquet` |
| 1 | Data layer + offline TasteNet + eval | ✅ DONE (`99c9d5e`, `30de8ac`) | NDCG@10 val 0.496 / test 0.519; acceptance PASS |
| 2 | World + swarm + calibration + Learn | ✅ DONE (`1e7660f`, `e2bf316`) | Ladder PASS; parity 13/13; calib 7/12 |
| 3 | Composer generations | ✅ DONE (`e2bf316`) | `taste_composer` 138.27 (gap-closed 0.51) |
| 4 | Evolution loop | ✅ DONE (`2fe4565`) | `evolved` 149.54 (gap-closed 0.58) |
| 5 | Ship polish (figures, README, blog notes) | ✅ DONE (`2fe4565`) | 5 headline figures; README Mini-Feed; BLOG_NOTES.md |
| 6 | Baseline audit + reproduction | ✅ DONE | `notes/phase6.md`; fresh run == archived numbers |
| A | Bigger net + ranking loss + composer refresh + `demo.ipynb` | ✅ DONE | `notes/phaseA.md`; val +0.0045 CI>0; evolved 151.53; executed notebook |

**Current best ladder (10,000 bots × 7 days, seed 42, calibrated, 2,000 bootstraps; Phase-A ranker):**

| policy | IWE / bot | gap-closed vs oracle |
|---|---|---|
| random | 61.86 | 0.00 |
| popularity | 77.87 | 0.11 |
| personal_affinity | 93.64 | 0.21 |
| taste (learned ranker) | 120.26 | 0.39 |
| taste_composer | 140.44 | 0.52 |
| **evolved (Phase A)** | **151.53** | **0.59** |
| oracle (adaptive ceiling) | 213.10 | 1.00 |

(Phase-2/4 numbers with the original 32k ranker were taste 120.28, composer 138.27, evolved 149.54.)
All adjacent rungs CI-separated; `taste_composer > taste` +20.18 and `evolved > taste_composer`
+11.09 (both CIs exclude 0). Raw: `recommender-engine/artifacts/sim_results.json`.

---

## Phase 0 — Data prep  ✅

**Built:** `recommender-engine/prepare_data.py` (the one sanctioned raw reader), `feed/build.py`.
One-time `raw → artifacts/` with column selection + chunked IO.

- 19,610,588 events after dedup → k-core 5/3 (3 iters) **13,061,273 rows**, 25,865 users,
  1,303,360 items → `features_{train,val,test}.parquet` (2.61M / 1.96M / 1.96M rows, 28 features,
  0 NaNs), `user_interest.parquet`, `user_stats.parquet`, `item_stats.parquet`,
  `real_marginals.json`, `session_hours.json`, `feature_dictionary.csv`, `preprocess_config.json`,
  `schema.json`. Full run 90.6s.
- Configs: `config/policy.yaml`, `config/swarm.yaml`. Every deviation logged in `ASSUMPTIONS.md`.

**Left:** nothing.

## Phase 1 — Data layer + Taste model + offline eval  ✅

**Built:** `feed/data.py`, `feed/model.py`, `feed/eval.py`, `feed/train.py`, `run_offline.py`,
`tests/test_phase1.py`.

- Train-only scaler (log1p `i_n`, std clipped), masked heads (search rows mask like/follow/forward/
  watch), `models/taste.pt` (TasteNet, 32,357 params).
- Graded NDCG@10/@20 per user + user-bootstrap CIs + baselines; LightGBM ceiling as reference.
- **Acceptance PASS** val & test: taste 0.4961/0.5186 > b3 0.4766/0.4976 > b2 0.4822/0.5013 >
  b1 0.4293/0.4407; LightGBM ceiling 0.5213/0.5421. Report `notes/phase1.md`.

**Left:** nothing (offline model retained but not used in the sim ladder — see Phase 2).

## Phase 2 — World + swarm + calibration + Learn  ✅

**Built:** `feed/world.py`, `feed/content.py`, `feed/featurestate.py`, `feed/response.py`,
`feed/policies.py`, `feed/composer.py`, `feed/metrics.py`, `feed/swarm.py`, `feed/parity.py`,
`feed/learning.py`, `calibrate.py`, `run_swarm.py`, `run_learn.py`, `tests/test_phase2.py`.

- Lockstep Find→Rank→Compose→response→feedback simulator over one shared CRN world; user-conditional
  hidden appeal (the fix that gives personalization headroom, `ASSUMPTIONS A18`).
- **Calibration 7/12** targets within 15% (5 structural misses documented in `notes/calibration.md`).
- **Feature parity 13/13** replayable user-side features within 15% (gate PASS).
- **Learn stage** (`run_learn.py`): train the ranker on simulator logs → `models/taste_sim.pt`
  (val NDCG@10 0.493, by-bot split, no oracle distillation).
- **IWE ladder PASS**: random 61.86 < popularity 77.87 < personal_affinity 93.64 < taste 120.28 <
  oracle 213.10. Report `notes/phase2.md`; the pre-fix PARTIAL failure is preserved in its
  History note.

**Left:** nothing.

## Phase 3 — Composer generations  ✅

**Built:** `feed/composer.py` (seen penalty, category-run penalty, session affinity, cold bonus,
ε-exploration), `run_generations.py`, `tests/test_phase3.py`.

- Additive generations gated on paired-bootstrap IWE gain (CI excludes 0) **and** no guardrail
  degrading >10%. **KEEP** diversity `alpha` (+4.45) and session adaptation `eta` (+8.98);
  **REJECT** seen-penalty (−0.01), cold bonus (−1.04), ε-exploration (−2.34, also tripped a guardrail).
- `taste_composer` = **138.27** (gap-closed 0.51, +13.4 over neutral), improving every guardrail.
  Reports `notes/phase3.md`, `notes/GENERATIONS.md`.

**Left:** nothing.

## Phase 4 — Evolution loop  ✅

**Built:** `feed/evolve.py` ((1+λ)-ES mechanics), `run_evolve.py` (entry),
`feed/policies.py:effective_policy` + `evolved` rung wired through `swarm.py` / `run_swarm.py`,
`config/policy_evolved.yaml`, `tests/test_phase4.py`.

- 16 knobs (7 composer + 5 score weights + 4 retrieval counts); σ = 15% of range decaying ×0.85;
  fitness = IWE − 5%·IWE per guardrail degraded >10%. Search TRAIN (2k×3d seed 1) → select VAL
  (5k×7d seed 2) → TEST once (10k×7d seed 3); 8-worker `multiprocessing`; 900s budget.
- Search monotone 58.08 → 62.11 (TRAIN); VAL best 149.84; TEST **150.03 vs 136.50**
  (+13.53 [+12.10, +14.92]).
- Seed-42 ladder `evolved` = **149.54** (gap-closed 0.58, +11.27 [+9.94, +12.74]); guardrails improve
  (coverage 78.5→81.7%, entropy 3.47→3.65 bits, repeat 0.54→0.52%). Evolve 335s, ladder 130s.
- Report `notes/phase4.md`; `ASSUMPTIONS A21`.

**Left:** nothing.

## Phase 5 — Ship polish  ✅

**Built:** `feed/plotting.py` (figure module, palette/dpi from `swarm.yaml:figures`, renders from
the committed artifacts only — no sim re-runs), five headline figures in `notes/figures/`,
root `README.md` **"Mini-Feed"** section, `notes/BLOG_NOTES.md`, the ten-line summary above.

- `01_ladder.png` — THE LADDER: % oracle gap closed per policy with CIs propagated from the
  adjacent paired-bootstrap diffs (quadrature); bold gap %, italic IWE/bot.
- `02_generations.png` — composer generations with KEEP/REJECT verdicts and the accepted-IWE
  trajectory (124.77 → 129.23 → 138.21).
- `03_calibration.png` — 12 targets sorted by |rel err| with the 15% band; 7 PASS / 5 structural
  misses.
- `04_cohorts.png` — per-cohort gap-closed heatmap (A=activity tercile × B=breadth tercile);
  personalization headroom concentrates in heavy users (A2_B0 61% → 76% under `evolved`).
- `05_retention.png` — return curves, D7: random 45.9 → evolved 56.9 → oracle 67.1%.
- Root README: one-line hook + mermaid **Find → Rank → Compose → Learn** + THE LADDER chart+
  table, 3-command quickstart (`run_offline.py`, `run_swarm.py`, `run_evolve.py`),
  "What didn't work" (γ −0.01, β −1.04, ε −2.34 + guardrail trip, 5 calibration misses) and the
  relative-claims honesty clause; stale Roadmap/Structure/Status sections updated.
- `notes/BLOG_NOTES.md`: 10 story beats, each with real numbers + honesty beats appendix.

**Left:** nothing.

## Phase 6 — Baseline audit + reproduction  ✅

**Built:** `feed/fingerprint.py` (split/config sha256 fingerprint), `notes/phase6.md`.

- Audited the shipped model for weaknesses; reproduced the Phase-1 baseline **exactly** on a fresh
  run (n_params 32,357, best epoch 5, val 0.496055, test 0.518649, identical per-epoch history).
- Archived the original artifact as `offline_results_phase1.json` before overwriting.

**Left:** nothing.

## Phase A — Bigger network, better objective, better composer  ✅

**Built:** config-driven variants in `feed/model.py` (+`feed/losses.py` L0–L7, hybrid ranking loss in
`feed/train.py`), improved composer components in `feed/composer.py`, drivers `run_capacity.py` /
`run_losses.py` / `run_improve.py`, executed `demo.ipynb`.

- **A1/A2.** `wide_hybrid` (122,773 params, masked BCE + LambdaRank surrogate) beats the new-code
  baseline on val **+0.0045 [+0.0028, +0.0062]** and test **+0.0049**, and clears the archived
  Phase-1 numbers. `notes/results/{improve,capacity,loss_ablation}.json`.
- **A3.** Composer additions (recency-weighted + length-adaptive diversity, now in the evolution
  spec). Greedy growth `120.26 → 140.54`; evolution **TEST 150.16 vs 140.22** (+9.95
  [+8.71, +11.25]); ladder `taste 120.26 → taste_composer 140.44 → evolved 151.53`. Reports
  `notes/phaseA.md`, `notes/results/{composer,evolve,sim}.json`.
- **A4.** Dataset fingerprint, paired user-bootstrap CIs, one-touch test, parity 13/13, 35 green tests.

**Left:** nothing.

---

## Repository layout

```
recommender-engine/
  prepare_data.py      p0  raw -> artifacts/ (one-time)
  feed/build.py        p0  clean / k-core / as-of features
  feed/data.py         p1  loaders, train-only scaler
  feed/model.py        p1  TasteNet
  feed/eval.py         p1  NDCG, bootstrap, baselines
  feed/train.py        p1  training loop
  run_offline.py       p1  entry: --fast / full
  feed/world.py        p2  catalog + bots (user-conditional appeal)
  feed/content.py      p2  latent content space
  feed/featurestate.py p2  incremental live features + parity replay
  feed/response.py     p2  response / leave / return psychology
  feed/policies.py     p2  retrieval pools + policy rankers (+ evolved overlay)
  feed/composer.py     p3  Compose re-ranker
  feed/metrics.py      p2  IWE, cohorts, guardrails
  feed/swarm.py        p2  lockstep simulator
  feed/parity.py       p2  feature-parity gate
  feed/learning.py     p2  Generate -> Train logs
  run_learn.py         p2  train sim-native ranker
  calibrate.py         p2  fit constants -> calibration.json
  run_swarm.py         p2  ladder entry
  run_generations.py   p3  composer generations
  feed/evolve.py       p4  (1+lambda)-ES mechanics
  run_evolve.py        p4  evolution entry
  feed/plotting.py     p5  the five headline figures
  feed/losses.py       pA  L0-L7 objectives + calibration diagnostics
  feed/experiment.py   pA  shared train+eval for ablations
  feed/fingerprint.py  p6  dataset/config fingerprint
  run_capacity.py      pA  network-capacity study
  run_losses.py        pA  loss ablation + diagnostics figure
  run_improve.py       pA  headline: baseline vs improved, val+test once
  build_demo.py        pB  generates demo.ipynb
  tests/               p1-pB  35 fast tests
  config/{policy,swarm,policy_evolved}.yaml
  ASSUMPTIONS.md       every modelling decision (A0-A21)
notes/                phase1-6 + phaseA write-ups, results/, figures/, GENERATIONS, BLOG_NOTES
demo.ipynb            executed end-to-end research notebook (repo root)
```

## Run / reproduce

```bash
cd recommender-engine
P=~/miniconda3/envs/picograd/bin/python
$P -m pytest tests/ -q          # 35 tests
$P run_offline.py --fast        # offline TasteNet smoke
$P run_improve.py --test baseline_masked,wide_hybrid   # A1/A2 headline (~700 s)
$P run_learn.py --variant wide --loss hybrid           # sim-native ranker
$P run_swarm.py                 # policy ladder incl. evolved (~130 s)
$P run_generations.py           # composer generations (~440 s)
$P run_evolve.py                # (1+lambda)-ES (~610 s) -> config/policy_evolved.yaml
$P -m feed.plotting             # the five headline figures -> ../notes/figures/
$P build_demo.py                # generate demo.ipynb, then nbconvert --execute it
```

## Known limitations / honest notes

- The simulator is **hand-designed and calibrated to marginals only** (7/12 targets); absolute IWE
  levels are not claims about real users — results are **relative** under the stated assumptions.
- The world was deliberately fixed to reward preference-alignment (`ASSUMPTIONS A18`); popularity
  is a weak rung here.
- The `oracle` is an upper bound, not deployable; its high repeat-exposure is a greedy artifact.
- Phase-4 fitness and IWE coincided (no guardrail violations during search), so the penalty acted as
  a gate that stayed healthy rather than a discriminator.
