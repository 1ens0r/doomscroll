# Build status — phase by phase

A from-scratch, CPU-only short-video recommender on **KuaiSAR-v2**: raw data → offline model →
calibrated behaviour simulator → policy ladder → staged **Compose** re-ranker → **evolution loop**.
Every policy claim is backed by a paired per-bot bootstrap CI, a guardrail panel, and a reproducible
negative result. This file tracks what is **built** vs **left**.

_Last updated: Phase 4 complete. All of Phases 0–4 committed and pushed; Phase 5 (ship polish) remains._

## At a glance

| Phase | Deliverable | State | Headline evidence |
|---|---|---|---|
| 0 | Data prep: raw → `artifacts/` | ✅ DONE (`2023ba8`) | 19.6M events → 13.06M k-core rows; `artifacts/*.parquet` |
| 1 | Data layer + offline TasteNet + eval | ✅ DONE (`99c9d5e`, `30de8ac`) | NDCG@10 val 0.496 / test 0.519; acceptance PASS |
| 2 | World + swarm + calibration + Learn | ✅ DONE (`1e7660f`, `e2bf316`) | Ladder PASS; parity 13/13; calib 7/12 |
| 3 | Composer generations | ✅ DONE (`e2bf316`) | `taste_composer` 138.27 (gap-closed 0.51) |
| 4 | Evolution loop | ✅ DONE (`2fe4565`) | `evolved` 149.54 (gap-closed 0.58) |
| 5 | Ship polish (README, figures, blog) | 🔲 NOT STARTED | — |

**Current best ladder (10,000 bots × 7 days, seed 42, calibrated, 2,000 bootstraps):**

| policy | IWE / bot | gap-closed vs oracle |
|---|---|---|
| random | 61.86 | 0.00 |
| popularity | 77.87 | 0.11 |
| personal_affinity | 93.64 | 0.21 |
| taste (learned ranker) | 120.28 | 0.39 |
| taste_composer | 138.27 | 0.51 |
| **evolved (Phase 4)** | **149.54** | **0.58** |
| oracle (adaptive ceiling) | 213.10 | 1.00 |

All adjacent rungs CI-separated; `taste_composer > taste` +17.99 and `evolved > taste_composer`
+11.27 (both CIs exclude 0). Raw: `recommender-engine/artifacts/sim_results.json`.

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
  b1 0.4293/0.4407; LightGBM ceiling 0.5213/0.5421. Report `reports/phase1.md`.

**Left:** nothing (offline model retained but not used in the sim ladder — see Phase 2).

## Phase 2 — World + swarm + calibration + Learn  ✅

**Built:** `feed/world.py`, `feed/content.py`, `feed/featurestate.py`, `feed/response.py`,
`feed/policies.py`, `feed/composer.py`, `feed/metrics.py`, `feed/swarm.py`, `feed/parity.py`,
`feed/learning.py`, `calibrate.py`, `run_swarm.py`, `run_learn.py`, `tests/test_phase2.py`.

- Lockstep Find→Rank→Compose→response→feedback simulator over one shared CRN world; user-conditional
  hidden appeal (the fix that gives personalization headroom, `ASSUMPTIONS A18`).
- **Calibration 7/12** targets within 15% (5 structural misses documented in `reports/calibration.md`).
- **Feature parity 13/13** replayable user-side features within 15% (gate PASS).
- **Learn stage** (`run_learn.py`): train the ranker on simulator logs → `models/taste_sim.pt`
  (val NDCG@10 0.493, by-bot split, no oracle distillation).
- **IWE ladder PASS**: random 61.86 < popularity 77.87 < personal_affinity 93.64 < taste 120.28 <
  oracle 213.10. Report `reports/phase2.md`; postmortem of the pre-fix failure in
  `reports/phase2_postmortem.md`.

**Left:** nothing.

## Phase 3 — Composer generations  ✅

**Built:** `feed/composer.py` (seen penalty, category-run penalty, session affinity, cold bonus,
ε-exploration), `run_generations.py`, `tests/test_phase3.py`.

- Additive generations gated on paired-bootstrap IWE gain (CI excludes 0) **and** no guardrail
  degrading >10%. **KEEP** diversity `alpha` (+4.45) and session adaptation `eta` (+8.98);
  **REJECT** seen-penalty (−0.01), cold bonus (−1.04), ε-exploration (−2.34, also tripped a guardrail).
- `taste_composer` = **138.27** (gap-closed 0.51, +13.4 over neutral), improving every guardrail.
  Reports `reports/phase3.md`, `reports/GENERATIONS.md`.

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
- Report `reports/phase4.md`; `ASSUMPTIONS A21`.

**Left:** nothing.

## Phase 5 — Ship polish  🔲 REMAINING

Not started. Planned deliverables (from the working plan):

- [ ] Root `README.md` section **"Mini-Feed"**: one-line hook + mermaid
      **Find → Rank → Compose → Learn** + THE LADDER chart.
- [ ] Ladder chart: random → popularity → personal → taste → composer → evolved → oracle, y = %
      oracle gap closed, with CIs → `reports/figures/*.png` (matplotlib, palette in
      `swarm.yaml:figures`).
- [ ] Generations timeline, calibration table, cohort heatmap, session-survival curve figures.
- [ ] 3-command quickstart (`run_offline.py`, `run_swarm.py`, `run_evolve.py`) in root README.
- [ ] "What didn't work" + limitations (hand-designed sim calibrated to marginals only; relative
      claims only) section.
- [ ] `reports/BLOG_NOTES.md`: 10 story beats, each with real numbers.
- [ ] `feed/plotting.py` figure module; (tests already exceed the 5-test target — **21 pass**).
- [ ] Final ≤10-line summary of the whole project.

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
  feed/plotting.py     p5  figures                [TODO]
  tests/               p1-p4  21 fast tests
  config/{policy,swarm,policy_evolved}.yaml
  ASSUMPTIONS.md       every modelling decision (A0-A21)
```

## Run / reproduce

```bash
cd recommender-engine
P=~/miniconda3/envs/picograd/bin/python
$P -m pytest tests/ -q          # 21 tests
$P run_offline.py --fast        # offline TasteNet smoke
$P run_swarm.py                 # policy ladder incl. evolved (~130 s)
$P run_generations.py           # composer generations (~209 s)
$P run_evolve.py                # (1+lambda)-ES (~335 s) -> config/policy_evolved.yaml
```

## Known limitations / honest notes

- The simulator is **hand-designed and calibrated to marginals only** (7/12 targets); absolute IWE
  levels are not claims about real users — results are **relative** under the stated assumptions.
- The world was deliberately fixed to reward preference-alignment (`ASSUMPTIONS A18`); popularity
  is a weak rung here.
- The `oracle` is an upper bound, not deployable; its high repeat-exposure is a greedy artifact.
- Phase-4 fitness and IWE coincided (no guardrail violations during search), so the penalty acted as
  a gate that stayed healthy rather than a discriminator.
