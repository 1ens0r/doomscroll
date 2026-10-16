# Phase 1 — Taste model + offline eval

**Verdict: ACCEPTANCE PASS** — Taste > B3 > B2 > B1 on val AND test (feed impressions,
graded NDCG@10), every adjacent paired-bootstrap 95% CI excludes 0. Test set touched once.

## What was built

| file | role |
|---|---|
| `feed/model.py` | `TasteNet` two-tower model (~32k params): user tower (9 `u_*` → 64 → 32), video tower (5 `i_*` + category embedding(38,16) → 64 → 32), pair block (6 `uc_*` + 2 `ui_*` + 2 `ctx_*` + 3 `cold_*` → 32), fusion concat `[user, video, user*video, pair]` (128) → 128 → 64 → 5 sigmoid heads |
| `feed/eval.py` | vectorized graded NDCG@10/@20 per user, eligibility (≥2 impressions, ≥2 distinct tiers), user-level bootstrap CIs (2000 resamples), paired diffs, baselines |
| `feed/train.py` | masked per-head BCE, `pos_weight = min(sqrt(neg/pos), 10)` from train, loss weights 1/2/3/3/1, AdamW 2e-3 + OneCycle, ≤6 epochs, early stop on val NDCG@10 |
| `run_offline.py` | `--fast` smoke / full run, ladder vs baselines, paired CIs, acceptance gate, optional LightGBM lambdarank ceiling, writes `artifacts/offline_results.json` |
| `tests/test_phase1.py` | 5 fast tests: NDCG toy math, search-row masks, train-only scaler, pos-weight cap, eligibility + bootstrap |

Score for ranking = `1*click + 2*watch + 3*like + 4*forward + 5*follow` (predicted probabilities,
weights from `config/policy.yaml`). Search rows mask like/follow/forward/watch in training only
(they are unobserved, not negative — ASSUMPTIONS A2).

## Ladder — graded NDCG over each user's label-window feed impressions

Gains `[0,1,2,3,4]` by tier. B1 = random, B2 = global engagement popularity (`i_mean_score`),
B3 = personalized popularity (`i_mean_score × (1 + uc_lift)`). `diag_*` rows are the plan's
literal exposure-based scores, kept for transparency (see Findings).

### val — feed (primary)

| method | ndcg@10 | 95% CI | ndcg@20 | 95% CI |
|---|---|---|---|---|
| taste | 0.4961 | [0.4907, 0.5016] | 0.5506 | [0.5456, 0.5556] |
| b1_random | 0.4293 | [0.4241, 0.4348] | 0.4835 | [0.4788, 0.4887] |
| b2_popularity | 0.4766 | [0.4714, 0.4823] | 0.5323 | [0.5275, 0.5374] |
| b3_affinity | 0.4822 | [0.4770, 0.4879] | 0.5372 | [0.5324, 0.5424] |
| diag_exposure_pop (`i_n`) | 0.4167 | [0.4115, 0.4221] | 0.4743 | [0.4694, 0.4794] |
| diag_decay_share (`uc_decay_share`) | 0.4249 | [0.4196, 0.4302] | 0.4894 | [0.4844, 0.4944] |

### test — feed (primary)

| method | ndcg@10 | 95% CI | ndcg@20 | 95% CI |
|---|---|---|---|---|
| taste | 0.5186 | [0.5139, 0.5234] | 0.5714 | [0.5671, 0.5756] |
| b1_random | 0.4407 | [0.4356, 0.4455] | 0.4973 | [0.4927, 0.5017] |
| b2_popularity | 0.4976 | [0.4930, 0.5027] | 0.5530 | [0.5485, 0.5578] |
| b3_affinity | 0.5013 | [0.4967, 0.5062] | 0.5566 | [0.5523, 0.5611] |
| diag_exposure_pop (`i_n`) | 0.4326 | [0.4280, 0.4376] | 0.4907 | [0.4862, 0.4954] |
| diag_decay_share (`uc_decay_share`) | 0.4442 | [0.4394, 0.4492] | 0.5067 | [0.5021, 0.5116] |

### Paired bootstrap diffs (feed, val / test) — acceptance evidence

| pair | val ndcg@10 | test ndcg@10 | verdict |
|---|---|---|---|
| taste − b3 | +0.0138 [+0.0109, +0.0168] | +0.0173 [+0.0144, +0.0202] | PASS |
| b3 − b2 | +0.0056 [+0.0034, +0.0077] | +0.0037 [+0.0017, +0.0059] | PASS |
| b2 − b1 | +0.0473 [+0.0435, +0.0510] | +0.0569 [+0.0532, +0.0606] | PASS |

Same three pairs also pass at ndcg@20 on both splits (`artifacts/offline_results.json`).

### Reported separately (not part of acceptance)

- **search scope** (is_search==1): taste 0.4049 val / 0.4303 test — beats random and B3, but sits
  slightly under B2 popularity (0.4098 / 0.4375). Search rows only log clicks; the graded ranking
  task there is thin (tier 2 vs 0), so the taste model adds little over popularity.
- **cold items** (feed ∧ `i_n == 0`, ~65% of feed rows): taste 0.6870 val / 0.6954 test, beats all.
  B2/B3/exposure collapse to the same 0-score ordering by construction (cold items have no prior
  stats), so their identical NDCGs are expected, not a bug.
- **LightGBM lambdarank ceiling** (identical features, 300 trees, 50s, best iter 295/300):
  feed 0.5213 val / 0.5421 test, search 0.4429 / 0.4657, cold 0.7030 / 0.7148. Beats Taste by
  ~+0.025 ndcg@10 — the expected GBDT-on-tabular edge; kept as an upper reference for Phase 2+.

## Findings (blog material)

1. **Exposure counters anti-rank within-user.** Ranking a user's own impressions by prior
   impression count (`i_n`) *loses* to random (0.4167 vs 0.4293 val @10), and `uc_decay_share`
   ties it (0.4249). This is not a bug: it is exposure/repeat bias, exactly what the EDA warned
   ("popularity is a coverage hack, not a ranker"). Repeat decay 52%→15%→10.6%→4.1% is visible
   in-feature. The offline ladder therefore uses *engagement-based* popularity (B2) and
   *quality × affinity* personalization (B3), with the literal exposure scores reported as
   diagnostics. Decision recorded in ASSUMPTIONS A13.
2. **What ranks well within-user**: item quality priors (`i_play_ratio` 0.4815, `i_mean_score`
   0.4766, `i_pos_rate` 0.4745) and category exposure share (0.4421). Habit rates are
   near-constant within a user and carry no within-user ranking signal.
3. **The model learns a blend**: 32k params, 5 productive epochs, val NDCG@10 0.4901 → 0.4961;
   +0.014 over B3 with CI excluding 0. Upside remains for the LightGBM-class ceiling.
4. **Cold-start is the hard part**: cold rows are 65% of feed impressions and rank much higher
   (0.687 vs 0.496) because their label mix is milder; the model still beats baselines there.

## Data-layer fixes found while building

- `feed/data.py:load_split` requested `watch` twice (`HEADS` + explicit) and `cold_item`/
  `cold_user` twice (`FEATURES` + `EVAL_COLS`); pyarrow returns duplicate columns, which broke
  `load_xy` downstream. Fixed with a de-dup pass at read time (order-preserving).
- The committed `artifacts/scaler.json` was stale (27 feature names but 29 mean/std entries —
  likely hand-edited). Rebuilt by re-running `python -m feed.data` (train-only, 2,612,256 rows,
  27 dense features). Artifacts are gitignored and regenerated.

## Timings (Ryzen 5700G, 8 threads, CPU)

| stage | full run | note |
|---|---|---|
| load (3 splits, 42 cols) | 4.7s | column-selective parquet |
| train (5 productive epochs, 2.61M rows, batch 4096) | 112.2s | ~19s/epoch + val eval, 6/6 epochs run |
| predict val+test (3.9M rows) | 4.5s | batched 65,536 |
| eval + bootstrap (6 scores × 3 scopes × 2 splits, 2000 resamples) | 20.1s | vectorized |
| LightGBM ceiling | 51.4s | budget 600s, fitted once |
| total | 205.3s (incl. LGBM); 152.4s without; 90.8s with `--reuse` + LGBM | |

`run_offline.py --fast` smoke: 12s end-to-end (200k/300k/200k row samples, 1 epoch, 500 boot).

## Reproduce

```bash
cd recommender-engine
~/miniconda3/envs/picograd/bin/python run_offline.py --fast   # smoke, ~12s
~/miniconda3/envs/picograd/bin/python run_offline.py          # full + LGBM ceiling
~/miniconda3/envs/picograd/bin/python -m pytest tests/test_phase1.py -q
```

Outputs: `models/taste.pt` (best checkpoint + metadata), `artifacts/offline_results.json`
(ladder, CIs, acceptance, timings), `artifacts/offline_results_fast.json` (smoke).
