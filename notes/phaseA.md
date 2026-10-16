# Phase A — bigger network, better objective, better composer

Continues `notes/phase6.md` (baseline audit). Everything here is measured **in-process against the
new-code baseline** (the refactor that routes L0 through `feed/losses.py` shifts per-epoch float
drift ≤0.3% but leaves peak val NDCG@10 unchanged, so comparisons are apples-to-apples). Test is
touched **once** per arm, after selection on validation.

Seed 42, CPU, 8 torch threads. Headline artifacts: `notes/results/{improve,capacity,loss_ablation,
composer,evolve,sim,loss_diagnostics}.json`.

## A1/A2 — bigger network + ranking-aware objective

`run_improve.py --test baseline_masked,wide_hybrid`:

| config | params | best ep | val NDCG@10 | val NDCG@20 | test NDCG@10 | test NDCG@20 |
|---|---|---|---|---|---|---|
| `baseline_masked` (L0 BCE, 32k) | 32,357 | 3 | 0.4961 | 0.5497 | 0.51776 | 0.57080 |
| **`wide_hybrid`** (122k, hybrid) | **122,773** | 6 | **0.5006** | **0.5544** | **0.52266** | **0.57550** |

- **val** paired difference **+0.0045, 95% CI [+0.0028, +0.0062]** (excludes 0).
- **test** difference **+0.0049** (0.52266 − 0.51776).
- Both clear the archived Phase-1 baseline (val 0.496055, test 0.518649).

The winner is selected on val with paired-bootstrap CIs; `within_ci = [wide_hybrid]`. Per-epoch
history (winner): train loss `3.8762 → 2.8607`, val NDCG@10 `0.4931 → 0.5006`, val loss
`2.5102 → 2.3972`; no early stop (improved every epoch). Full history in `notes/results/improve.json`.

### Capacity alone is neutral (`run_capacity.py`, full)

| variant | params | best ep | val NDCG@10 | val NDCG@20 | train s | Δ vs baseline [95% CI] |
|---|---|---|---|---|---|---|
| baseline | 32,357 | 3 | 0.4961 | 0.5497 | 352 | (reference) |
| wide | 122,773 | 3 | 0.4949 | 0.5484 | 491 | −0.0012 [−0.0028, +0.0003] |
| wide_deep | 104,181 | 4 | 0.4946 | 0.5485 | 564 | −0.0015 [−0.0032, +0.0002] |
| cross | 36,965 | 3 | 0.4944 | 0.5486 | 329 | −0.0017 [−0.0032, −0.0001] |

Promoted = **baseline** (all CIs within noise; smallest params). Adding parameters under the plain
masked-BCE objective buys **nothing** — the win in A1/A2 comes from the **ranking-aware objective**,
not raw capacity. This is the central Phase-A finding: `wide` alone ≈ baseline, `wide + hybrid` beats it.

**Objective.** `wide_hybrid` = `feed.losses.HybridLoss`: masked per-head BCE (as L0) **plus** a
ranking-aware term — a differentiable LambdaRank surrogate on graded gains `[0,1,2,3,4]` over each
user's batch, mixed with a pairwise hinge at `train.rank_weight` (default 0.5). Because the capacity
study above shows `wide` ≈ `baseline` under plain BCE, the val/test gain of `wide_hybrid` is
attributable to the **objective**, not the parameters.

The objective ranking is confirmed on the **full-data 8-arm sweep** in `loss_ablation.json`
(baseline architecture, all train rows, shared protocol/seed; paired user-bootstraps vs `L0`):

| arm | change | best ep | val NDCG@10 | Δ vs L0 (95% CI) |
|---|---|---|---|---|
| `L3_lambdarank` | `train.loss=lambdarank` | 6 | **0.5119** | **+0.0158 [+0.0132, +0.0184]** |
| `L2_no_weighting` | `pos_weight_mode=none` | 4 | 0.5002 | +0.0041 [+0.0021, +0.0062] |
| `L2_cap5` | `pos_weight_cap=5` | 4 | 0.4974 | +0.0013 [-0.0003, +0.0029] |
| `L4_equal_heads` | `train.loss=equal_heads` | 4 | 0.4962 | +0.0001 [-0.0015, +0.0017] |
| `L5_cosine` | `train.scheduler=cosine` | 4 | 0.4961 | +0.0000 [-0.0016, +0.0016] |
| `L0_masked_bce` | reference | 3 | 0.4961 | — |
| `L4_uncertainty` | `train.loss=uncertainty` | 4 | 0.4945 | -0.0016 [-0.0033, -0.0001] |
| `L5_lr_5e-3` | `train.lr=5e-3` | 1 | 0.4932 | -0.0029 [-0.0049, -0.0011] |

Ranking-aware and reweighting changes lead; the ranking term (`L3_lambdarank`) is decisive. Removing
`sqrt` positive weighting helps a little, while an over-aggressive LR (`5e-3`) or learned uncertainty
weighting **hurt**. Per-head loss/calibration diagnostics (val BCE, Brier, log-loss, ECE, grad-norm)
are in `loss_diagnostics.json` → figure `06_loss_diagnostics.png`.

## A3 — better composer and evolution ladder

The Compose stage gained two components (`feed/composer.py`): **recency-weighted diversity**
(`alpha_recency`, weight recent category hits harder than the top of the window) and
**length-adaptive diversity** (`alpha_growth`, stiffen the diversity penalty as a session lengthens).
The evolution spec (`feed/evolve.py` + `config/policy.yaml`) now searches both (`r`, `g`). The ranker
driving the whole loop is the improved `wide_hybrid`, retrained on simulator logs
(`run_learn.py --variant wide --loss hybrid`, val NDCG@10 0.5423).

### Greedy component growth (`run_generations.py`, full)

| gen | component | knob=value | IWE | Δ vs accepted [95% CI] | guardrails | verdict |
|---|---|---|---|---|---|---|
| 0 | neutral composer | — | 120.26 | — | — | — |
| 1 | seen-penalty | gamma=2.0 | 120.43 | +0.17 [−0.57, +0.94] | ok | REJECT (no gain) |
| 2 | diversity | alpha=0.4 | 139.45 | **+19.19 [+17.73, +20.67]** | ok | **KEEP** |
| 3 | session adaptation | eta=0.5 | 140.44 | +0.99 [−0.24, +2.18] | BAD | REJECT (guardrail) |
| 4 | cold bonus | beta=0.3 | 137.64 | −1.80 [−2.67, −0.95] | BAD | REJECT |
| 5 | epsilon exploration | epsilon=0.05 | 137.58 | −1.87 [−2.53, −1.27] | ok | REJECT |
| 6 | recency-weighted diversity | alpha_recency=0.85 | 140.75 | +1.31 [+0.23, +2.36] | BAD | REJECT (guardrail) |
| 7 | length-adaptive diversity | alpha_growth=0.2 | 133.75 | −5.70 [−6.70, −4.69] | ok | REJECT |
| 8 | recency-boosted diversity (joint) | alpha=0.5, alpha_recency=0.85 | **140.54** | **+1.10 [+0.16, +2.04]** | ok | **KEEP** |

Accepted knobs: `{alpha: 0.5, alpha_recency: 0.85, ...}`. Neutral `120.26 → 140.54 (+20.29)`.

### Evolved ladder (`run_evolve.py` + `run_swarm.py`, full)

Evolution selected `{gamma 0.96, alpha 0.86, eta 1.20, decay 0.75, window_m 5,
alpha_recency 0.98, alpha_growth 0.089, score-weights…, pool_size 176}`. **VAL** best IWE 151.17;
**TEST** (seed 3, 10k×7) evolved **150.16** vs base composer **140.22**, Δ **+9.95 [+8.71, +11.25]**.

Full 7-policy ladder (10,000 bots × 7d; all adjacent paired CIs exclude 0):

| policy | IWE/bot | gap-closed |
|---|---|---|
| random | 61.86 | 0.00 |
| popularity | 77.87 | 0.11 |
| personal_affinity | 93.64 | 0.21 |
| taste | 120.26 | 0.39 |
| **taste_composer** | **140.44** | 0.52 |
| **evolved** | **151.53** | 0.59 |
| oracle | 213.10 | 1.00 |

Parity replay 13/13 user-side feature counters match. Versus Phase-4 (baseline ranker):
`taste_composer 138.27 → 140.44`, `evolved 149.54 → 151.53` — the acceptance bar (≥149.54) is met.

## A4 — methods and reproducibility

- **Dataset fingerprint** (`feed/fingerprint.py` → `notes/results/fingerprint.json`): sha256 per
  split + config hashes; train `93f9ca78…`, val `2d1345b7…`, test `897b6a57…`.
- **Selection discipline**: search on val, paired user-bootstrap CIs (2000 resamples), test touched
  once. All candidate arms share the identical data, split, candidate protocol and seed.
- **Leakage**: features are as-of the impression timestamp; dense scaling uses train-only stats; the
  simulator's serving-path check (`feed/parity.py`) re-derives user counters and matches 13/13.
- **Tests**: 35 green (`pytest`), including the new loss/variant/composer-generation shape tests.

## Negative results (kept, not hidden)

- Standalone **recency-weighted diversity** and **length-adaptive diversity** did not survive the
  greedy gate (guardrail violation / IWE drop); only the joint `{alpha 0.5, recency 0.85}` retune
  helped. Evolution still exploited both knobs at small values.
- Phase-3 rejects recur (seen-penalty, session adaptation, cold bonus, epsilon exploration).

## Reproduce

```bash
cd recommender-engine
P=~/miniconda3/envs/picograd/bin/python
$P run_improve.py --test baseline_masked,wide_hybrid     # A1/A2 (~700 s)
$P run_capacity.py                                        # A1 capacity
$P run_losses.py                                          # A2 loss ablation + diagnostics
$P run_learn.py --variant wide --loss hybrid              # sim ranker
$P run_generations.py && $P run_evolve.py && $P run_swarm.py
$P build_demo.py && python -m nbconvert --to notebook --execute ../demo.ipynb --output ../demo.ipynb
```
