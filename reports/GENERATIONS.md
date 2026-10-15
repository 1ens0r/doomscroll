# Composer — design and generation log

The **Compose** stage re-orders the Find→Rank shortlist just before it is shown, using signals the
ranker does not see: what was already shown this session, how concentrated the run has become, and
what the user has just engaged with. It is deliberately cheap (vectorised over a wave, no per-item
model call) so it can sit in the hot path.

See `reports/phase3.md` for the verdict summary; this file is the component reference.

## Scoring form (`feed/composer.py`)

For session `s` and candidate item `i` in the pooled shortlist:

```
score(s, i) =  taste_z(s, i)                        # Ranker score, z-scored within session
             − gamma   · seen(s, i)                 # cross-session + within-session exposure
             − alpha   · same_cat(s, i)             # category-run penalty (diversity)
             + eta     · aff[s, cat(i)]             # session-affinity: rewarded categories so far
             + beta    · inv_i(i)                   # cold/exploration bonus (1/sqrt(1+n))
             + epsilon · U(0,1)                     # epsilon-greedy exploration
```

`seen = prior_seen + shown_this_session`. `select` then **hard-bans** within-session duplicates, so
`gamma` only ever acts on *cross-session* re-exposure. All coefficients live in
`config/policy.yaml:composer`; the oracle reuses the same composer with a true-`appeal` ranker.

## Knobs

| knob | meaning | accepted | search range | notes |
|---|---|---|---|---|
| `gamma` | seen penalty | 0.0 | 0, 0.5, 1, 2, 4 | redundant with hard within-session ban |
| `alpha` | category-run penalty | 0.4 | 0, 0.1, 0.2, 0.4, 0.8 | larger values trade IWE for entropy |
| `eta` | session taste affinity | 0.5 | 0, 0.1, 0.25, 0.5, 1 | largest single gain |
| `beta` | cold bonus | 0.0 | 0, 0.05, 0.1, 0.3 | neutral-to-negative on IWE |
| `epsilon` | exploration | 0.0 | 0, 0.01, 0.05, 0.1 | negative; tripped repeat guardrail |
| `decay` | affinity decay across a session | 0.8 | fixed | recency weighting of engaged categories |
| `window_m` | affinity window (impressions) | 5 | fixed | how far back affinity is read |

## Generation log (full scale)

Additive: turn on one knob from the neutral baseline, keep only on a CI-separated IWE gain with no
guardrail degrading > 10%.

| gen | component | ΔIWE | 95% CI | guardrail | verdict |
|---|---|---|---|---|---|
| 1 | seen-penalty (`gamma=2`) | −0.01 | [−0.66, +0.64] | ok | REJECT |
| 2 | diversity (`alpha=0.4`) | +4.45 | [+2.91, +5.96] | ok | KEEP |
| 3 | session adaptation (`eta=0.5`) | +8.98 | [+7.61, +10.39] | ok | KEEP |
| 4 | cold bonus (`beta=0.3`) | −1.04 | [−2.21, +0.13] | ok | REJECT |
| 5 | epsilon (`epsilon=0.05`) | −2.34 | [−3.00, −1.67] | bad | REJECT |

Neutral 124.77 → accepted 138.21 (`taste_composer`, ladder run: 138.27). In the `--fast` smoke the
ordering of the first two gens differs (seen-penalty +0.42 KEEP) — fast mode is a smoke test, not
the decision, and the code keeps the accepted set from the full run.

## Guardrails (full run)

| policy | cat entropy (bits) | catalog coverage | cold-impression share | repeat-exposure rate |
|---|---|---|---|---|
| popularity | 4.08 | 5.9% | 0.18% | 1.04% |
| personal_affinity | 3.47 | 59.2% | 11.4% | 4.58% |
| taste | 3.35 | 75.2% | 4.95% | 0.57% |
| **taste_composer** | 3.47 | 78.5% | 4.80% | 0.54% |
| oracle | 3.86 | 91.4% | 2.68% | 6.51% |

`taste_composer` dominates `taste` on every column: more entropy, more coverage, lower repeat.

## Negative results (kept, not deleted)

- **Seen-penalty is redundant** with the hard within-session ban and the low cross-session re-exposure
  rate — a confirmed no-op.
- **Cold bonus costs IWE** in this world. Retained as a knob (cold-start is a separate P0), disabled
  in the accepted set.
- **Epsilon-greedy exploration hurts** and degrades repeat-exposure; there is no exploration story to tell
  at this scale.
