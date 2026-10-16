# Phase 6 — Model audit and baseline reproduction

**Verdict: REPRODUCED EXACTLY.** A fresh full run of the shipped Phase-1 entry point
(`run_offline.py`, no flags) reproduced the archived baseline metric-for-metric: parameter count,
best epoch, val/test graded NDCG@10, per-head positive weights, and every per-epoch train loss /
val NDCG. The archived Phase-1 artifact was copied aside first (never overwritten).

## What was read (no guessing)

`feed/build.py`, `feed/data.py` (`FEATURES`/`DENSE`/`HEAD_MASKS`/`make_targets`/`fit_scaler`),
`feed/model.py` (`block_indices`, `TasteNet`, `predict_probs`), `feed/train.py`
(`compute_pos_weights`, `masked_head_loss`, `train_taste`), `feed/eval.py` (`per_user_ndcg`,
`bootstrap_ci`, `collect_paired`, `acceptance`, `baseline_scores`, `lgbm_ceiling`), `run_offline.py`,
`config/policy.yaml`, `artifacts/{offline_results,preprocess_config,real_marginals,schema}.json`,
`notes/phase1.md`, `ASSUMPTIONS.md`, `README.md`, `status.md`, `notes/*.md`.

## Dataset fingerprint

Saved to `notes/results/fingerprint.json`. Config hashes (`sha256` of the file bytes):

| artifact | sha256 |
|---|---|
| `preprocess_config.json` | `65d206c54d7f8f0574faa2337219c19d4db607f260d183873138f02a18424f61` |
| `scaler.json` | `1bfe7b58dd1ca5732a607a087cfb554f7bc58864e10711c919774c7731ee4bf4` |

Per-split manifest hash (sha256 over the concatenated `user_id,item_id,ts,tier,is_search` columns,
in stored row order):

| split | rows | users | items | ts_min | ts_max | sha256 |
|---|---|---|---|---|---|---|
| train | 2,612,256 | 13,269 | 568,189 | 1685438304222 | 1685765928396 | `93f9ca782e12bf7e9296f81d1cacc80cccf3451706ee82475397c406c907707e` |
| val | 1,959,189 | 11,218 | 487,026 | 1685765928461 | 1686037179623 | `2d1345b743c908ca69145f383028dff807b32a2005df2ab4a44e2183e8234c57` |
| test | 1,959,194 | 10,634 | 444,869 | 1686037181048 | 1686360599225 | `897b6a5798c9b9080481653eaf1f19381d6e42a18e1f91f9deb902b226aff5e4` |

Gains `[0,1,2,3,4]`; split cuts `[1685438304222, 1685765928461, 1686037181048]`.

## Task, features, architecture

- **Task:** per-candidate graded ranking. Prediction unit = one (user, item) impression row. Eval
  scope = feed impressions (`is_search == 0`); metric = per-user graded NDCG@10/@20 (`tier_gains`),
  eligibility ≥2 impressions and ≥2 distinct tiers, user-level bootstrap (2000 resamples, seed 42).
- **Inputs:** 27 dense features + 1 categorical (`cat_id`, 38 embedding rows). Dense groups: 9 `u_*`
  habit, 5 `i_*` item prior, 13 `uc_*`/`ui_*`/`ctx_*`/`cold_*` pair/context.
- **Architecture (`TasteNet`):** user tower (9→64→32), video tower (5 + 16-dim cat embedding →64→32),
  pair block (13→32), fusion concat `[user, video, user*video, pair]` (128) →128→64→5. **32,357
  parameters** (`n_params` in the artifact).
- **Targets:** 5 heads `[click, like, follow, forward, watch]`; search rows mask like/follow/forward/
  watch in the loss (unobserved, not negative — ASSUMPTIONS A2); `watch` = clipped play ratio.
- **Loss:** per-head masked BCE normalised per head by observed count, then weighted by
  `loss_weights {click 1, like 2, follow 3, forward 3, watch 1}`; `pos_weight = min(sqrt(neg/pos), 10)`
  computed on train observed rows only.
- **Optimizer:** AdamW lr 2e-3, weight decay 1e-4, batch 4096, OneCycle over `max_epochs × steps`,
  ≤6 epochs, early stop patience 2 on val NDCG@10. Seed 42 everywhere; scaler fit on train only.

## Reproduction run (fresh, `run_offline.py`, no flags)

| item | archived Phase-1 | fresh Phase-6 | match |
|---|---|---|---|
| n_params | 32,357 | 32,357 | exact |
| best epoch | 5 / 6 | 5 / 6 | exact |
| best val NDCG@10 | 0.496055 | 0.496055 | exact (4 dp, 6 dp equal) |
| test NDCG@10 (taste) | 0.518649 | 0.518649 | exact |
| pos_weight (click…watch) | 1.0049, 5.6697, 10.0, 10.0, 1.1559 | identical | exact |

Per-epoch history (identical in both runs; fresh-run seconds shown):

| epoch | train loss | val NDCG@10 | val NDCG@20 | s |
|---|---|---|---|---|
| 1 | 3.5001 | 0.4901 | 0.5442 | 17.7 |
| 2 | 2.4963 | 0.4907 | 0.5453 | 18.1 |
| 3 | 2.4669 | 0.4928 | 0.5472 | 17.7 |
| 4 | 2.4514 | 0.4955 | 0.5492 | 17.9 |
| 5 | 2.4404 | 0.4961 | 0.5506 | 17.9 |
| 6 | 2.4315 | 0.4952 | 0.5497 | 18.0 |

Runtime: load 5.5s, train 108.2s, predict 4.2s, eval 21.1s, **total 205.9s**. Acceptance gate PASS
(all adjacent ladder paired-bootstrap CIs exclude 0 on val and test). LightGBM lambdarank ceiling
0.5213 val / 0.5421 test. Hardware: Ryzen 5700G, 8 threads, CPU, 30GB RAM, no CUDA.
Env: Python 3.13.14, torch 2.13.0+cu130, numpy 2.5.1, pandas 3.0.5 (Linux).

> This run was captured **before** the Phase-A edits to `feed/model.py` / `feed/train.py`. Those
> edits re-express the same architecture (parameter count still 32,357) but route the L0 loss through
> `feed/losses.py`; the autograd graph differs by an elementwise multiply, which introduces ≤0.3%
> per-epoch float drift while the **peak val NDCG@10 is unchanged (0.4961)**. Phase-A comparisons are
> therefore all made *in-process* against a new-code baseline.

## Model audit

- **Strengths:** reproducible to the digit on this hardware; small (32k params) and fast; the graded
  NDCG target matches the graded label tier; masking is correct per source; all baselines beaten with
  separated CIs.
- **Weaknesses driving Phase A:** `follow`/`forward` positive weights saturate the cap (10.0) at
  prevalence 0.004/0.003, so those heads are effectively un-learned; 5 scalar towers see almost no
  explicit cross-feature interaction; the objective is pointwise BCE despite a *ranking* target; and
  capacity is tiny (a LightGBM lambdarank ceiling sits ~+0.025 NDCG@10 above it).

## Reproduce

```bash
cd recommender-engine
P=~/miniconda3/envs/picograd/bin/python
$P run_offline.py              # full baseline + LightGBM ceiling (~206 s)
$P run_offline.py --fast       # smoke (~12 s), not a benchmark
$P -c "from feed.fingerprint import dataset_fingerprint as f; import json; print(json.dumps(f(),indent=1))"
```

Archived baseline preserved at `artifacts/offline_results_phase1.json`; fresh run at
`artifacts/offline_results.json`.
