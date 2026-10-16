"""Build `demo.ipynb` (repo root) from the executed research artifacts.

The notebook is a presentation artifact: it loads real data, renders committed figures/results,
shows the improved network's actual source, and runs a small live simulation. This script only
generates the cell structure; execution + output capture happens with `nbconvert --execute`.
"""
from __future__ import annotations

from pathlib import Path

import nbformat as nbf

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "demo.ipynb"

CELLS: list[tuple[str, str]] = [
    ("md", """# doomscroll — an end-to-end feed ranking demo

**Question.** Given a stream of (user, video) impressions, can a small neural ranker plus a
taste-conditioned *composer* order a session to maximise a graded engagement score (IWE), and does
it beat popularity / affinity / oracle-free baselines?

**Target.** Per-user graded NDCG@10 over 5 action heads (tier gains `[0,1,2,3,4]`), plus IWE in a
closed-loop simulator.

**How to read this notebook.**

| Section | Uses | Nature |
|---|---|---|
| Data loading, EDA, task, features | real `features_*.parquet` logs | **real data** |
| Improved network, training loop, evaluation | trained checkpoints + committed results | **real numbers** |
| Live single-subject simulation | the closed-loop simulator | **simulated** (no causal claims) |

Seed 42 throughout. CPU only."""),

    ("md", "## 0. Setup — environment, dataset fingerprint, reproducibility"),
    ("code", """import json, platform, subprocess, sys
from pathlib import Path

import numpy as np
import torch

REPO = Path.cwd()
if not (REPO / "recommender-engine").exists():          # tolerate being run from inside the engine
    REPO = REPO.parent
ENGINE = REPO / "recommender-engine"
sys.path.insert(0, str(ENGINE))
torch.set_num_threads(8)

try:
    commit = subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"],
                                     text=True).strip()
except Exception:
    commit = "(no git)"

fp = json.loads((REPO / "notes" / "results" / "fingerprint.json").read_text())
print("doomscroll end-to-end demo")
print(f"python {platform.python_version()} | torch {torch.__version__} | device=cpu | seed=42 | git {commit}")
print("DISCLAIMER: offline metrics come from real logs; the live-simulation section is SIMULATED.")
print("\\ndataset fingerprint (sha256 per split):")
for s in ("train", "val", "test"):
    m = fp["splits"][s]
    print(f"  {s:<5} rows={m['n_rows']:>9,}  users={m['n_users']:>7,}  items={m['n_items']:>7,}"
          f"  {m['sha256'][:12]}…")"""),

    ("md", """## 1. Data loading

Real feature artifacts live in `recommender-engine/artifacts/features_{split}.parquet`. We always
read a **column subset** (never the full frame), and the split boundaries are time-based cuts so
val/test are strictly future data."""),
    ("code", """import pandas as pd
from feed.data import load_split, DENSE

sub = ["user_id", "item_id", "ts", "tier", "is_search", "cold_item", "i_mean_score", "uc_lift"]
frames = {}
for s in ("train", "val", "test"):
    df = load_split(s, columns=sub)
    frames[s] = df
    print(f"{s:<5} rows={len(df):>9,}  cols={df.shape[1]}  users={df['user_id'].nunique():>7,}"
          f"  search={df['is_search'].mean():.3f}  cold_item={df['cold_item'].mean():.3f}")
print("\\nhead of train:")
frames["train"].head(5)"""),

    ("md", "## 2. EDA — figures and result tables (rendered, not re-plotted)"),
    ("code", """from IPython.display import Image, Markdown, display

for p in sorted((REPO / "notes" / "figures").glob("*.png")):
    display(Markdown(f"### `notes/figures/{p.name}`"))
    display(Image(filename=str(p)))"""),
    ("code", """for name in ("capacity.json", "loss_ablation.json", "improve.json",
             "loss_diagnostics.json", "fingerprint.json"):
    p = REPO / "notes" / "results" / name
    display(Markdown(f"### `notes/results/{name}`"))
    txt = json.dumps(json.loads(p.read_text()), indent=1)
    print(txt[:1800] + ("\\n… (truncated)" if len(txt) > 1800 else ""))"""),

    ("md", """## 3. Task and labels

Five action heads. `click` is observed on every row; `like/follow/forward/watch` are only observed
on **feed** rows, so search rows are **masked**, not treated as negatives (ASSUMPTIONS A2). The
graded objective maps the engagement `tier` to gains `[0,1,2,3,4]`."""),
    ("code", """from feed.data import make_targets, HEADS

df = load_split("train", columns=["tier", "is_search", "click", "like", "follow", "forward",
                                  "watch", "obs_like", "obs_follow", "obs_forward", "obs_watch"])
gains = json.loads((ENGINE / "artifacts" / "preprocess_config.json").read_text())["tier_gains"]
y, m = make_targets(df)
print("tier gains:", gains)
print("tier distribution:", np.bincount(df["tier"].to_numpy(), minlength=5).tolist())
print("head positive rate (observed only):")
for j, h in enumerate(HEADS):
    rate = float((y[:, j] * m[:, j]).sum() / max(1.0, m[:, j].sum()))
    print(f"  {h:<8} pos={rate:.4f}  masked={1 - m[:, j].mean():.4f}")"""),

    ("md", """## 4. Features

27 dense features in three groups plus one categorical (`cat_id`), all constructed **as-of** the
impression timestamp to avoid leakage. Dense values are standardised with **train-only** statistics
(`log1p` on the heavy-tailed count `i_n`)."""),
    ("code", """from feed.data import load_scaler

sc = load_scaler()
meta = json.loads((ENGINE / "artifacts" / "feature_list.json").read_text())
groups = {
    "user habits (u_*)": [f for f in DENSE if f.startswith("u_")],
    "item priors (i_*)": [f for f in DENSE if f.startswith("i_")],
    "pair/context (uc_/ui_/cold_/ctx_*)": [f for f in DENSE if f.startswith(("uc_", "ui_", "cold_", "ctx_"))],
}
print(f"dense features: {len(DENSE)}  |  categorical: cat_id → {meta['cat_embed_rows']} embedding rows")
for name, fs in groups.items():
    print(f"  {name}: {len(fs)}  {fs}")
print(f"scaler fit_on={sc['fit_on']} n_rows={sc['n_rows']:,} log1p={sc['log1p']}")
print("leakage note: val/test are standardised with the train scaler only (no refit).")"""),

    ("md", """## 5. The improved network

The ranker is a config-driven `TasteNet` (two towers + a `u*v` interaction block). The source below
is the exact class shipped in `recommender-engine/feed/model.py` — mirrored here cell by cell."""),
    ("code", """import inspect
import yaml
from feed import model as M
from feed.model import build_model, list_variants, n_params, resolve_model_cfg

print(inspect.getsource(M.TasteNet))"""),
    ("code", """pol = yaml.safe_load((ENGINE / "config" / "policy.yaml").read_text())
print("named variants and their parameter counts:")
for v in list_variants(pol):
    cfg = resolve_model_cfg(pol, v)
    p = n_params(build_model(meta, pol, variant=v))
    print(f"  {v:<10} params={p:>8,}  cfg={cfg}")"""),

    ("md", "### Objective\nPointwise BCE on 5 heads, optionally mixed with a **ranking-aware** (LambdaRank + pairwise) term for the *improved* run."),
    ("code", """from feed import losses as L
print(inspect.getsource(L.HybridLoss))"""),

    ("md", "## 6. Training loop"),
    ("code", """from feed.train import train_taste
print(inspect.getsource(train_taste))"""),
    ("code", """imp = json.loads((REPO / "notes" / "results" / "improve.json").read_text())
promoted = imp.get("promoted") or imp.get("best")
print(f"promoted configuration: {promoted}  (best={imp.get('best')}, reference={imp.get('reference')})")
r = imp["results"][promoted]
print(f"{'epoch':>5} {'train_loss':>11} {'val_loss':>10} {'val_ndcg@10':>12} {'val_ndcg@20':>12} {'lr':>10}")
for i, h in enumerate(r["history"], 1):
    mark = "  <- best" if i == r["best_epoch"] else ""
    print(f"{i:>5} {h['train_loss']:>11.4f} {h['val_loss']:>10.4f} {h['val_ndcg10']:>12.4f}"
          f" {h['val_ndcg20']:>12.4f} {h['lr']:>10.2e}{mark}")
print(f"best_epoch={r['best_epoch']}  params={r['n_params']:,}  train_seconds={r['train_seconds']:.1f}")"""),

    ("md", "## 7. Evaluation — improved vs baselines (val + test)"),
    ("code", """print(f"{'config':<18}{'params':>9}{'best_ep':>8}{'val@10':>9}{'val@20':>9}")
for label, rr in imp["results"].items():
    v = rr["val_feed"]
    print(f"{label:<18}{rr['n_params']:>9,}{rr['best_epoch']:>8}"
          f"{v['ndcg@10']['mean']:>9.4f}{v['ndcg@20']['mean']:>9.4f}")
print("\\npaired-bootstrap difference vs reference (val NDCG@10):")
print(json.dumps(imp["paired_vs_reference"], indent=1))
if "test_feed" in imp:
    print("\\ntest NDCG (touched once):")
    for label, v in imp["test_feed"].items():
        print(f"  {label:<18} ndcg@10={v['ndcg@10']['mean']:.4f}  ndcg@20={v['ndcg@20']['mean']:.4f}")"""),
    ("code", """# cohort / calibration / cold-start diagnostics were produced by the simulator report and are
# rendered in section 2 (notes/figures/03_calibration.png, 04_cohorts.png, 05_retention.png).
print("See the rendered figures above for calibration, cohorts, and retention.")
print("Baselines beaten with separated CIs in the committed offline report "
      "(recommender-engine/artifacts/offline_results.json).")"""),

    ("md", """## 8. Live single-subject simulation

We drive the closed-loop simulator (`feed/swarm.py`) with the improved ranker and the
**taste-conditioned composer**, then print one subject's trace swipe by swipe. This is
**simulated** data — the responses come from the world model, not from logs."""),
    ("code", """from feed.policies import TasteScorer
from feed.world import load_config
from feed.swarm import simulate

cfg = load_config()
cfg["policy"] = pol
pol["_models_dir"] = ENGINE / "models"
label = imp.get("promoted") or imp.get("best")
pol["_taste_ckpt"] = f"improve/{label}.pt"
scorer = TasteScorer(pol, int(pol["python_threads"]))
print(f"live ranker checkpoint: models/improve/{label}.pt  (variant={scorer.variant})")

trace: list[dict] = []
stats = simulate("taste_composer", cfg, seed=42, scorer=scorer, n_bots=48, n_days=1,
                 trace=trace, log=lambda *a: None)
bot = min(t["bot"] for t in trace)
steps = [t for t in trace if t["bot"] == bot]
print(f"single-subject trace — subject #{bot}, {len(steps)} swipes (SIMULATED)")
prev = None
for i, t in enumerate(steps[:25], 1):
    cur = f"item {prev['item']:<6}" if prev else "— (session start)"
    acts = [k for k in ("click", "like", "follow", "forward") if t[k]]
    if t["watch"] > 0:
        acts.append(f"watch={t['watch']:.2f}")
    action = ", ".join(acts) if acts else "scroll past"
    print(f"[{i:02d}] currently watching: {cur} → next recommended: item {t['item']:<6}"
          f" (cat {t['cat']:>2}) → action: {action:<24} | tier {t['tier']}")
    prev = t
print(f"\\nsimulator summary: impressions={stats['impressions']:,} "
      f"iwe/bot={stats['iwe_mean']:.3f} (SIMULATED — no causal claim)")"""),

    ("md", """## 9. Conclusions and limitations

- A 32k→123k parameter `TasteNet` with a ranking-aware objective improves val/test NDCG@10 over the
  Phase-1 baseline; selection is on validation with a paired-bootstrap CI, and test is touched once.
- The **composer** (diversity, novelty, recency- and length-adaptive re-ranking) raises IWE over a
  pure taste ranker in the simulator; guardrails check it does not collapse relevance.
- **Limitations.** All engagement targets are observational; the `watch` head is a proxy; the
  simulator is a model of behaviour, so its IWE is not a causal effect. Search rows are masked
  rather than negative.

### Reproduce

```bash
cd recommender-engine
python run_offline.py        # baseline + ladder + LightGBM ceiling
python run_improve.py --test baseline_masked,wide_hybrid
python build_demo.py && jupyter nbconvert --to notebook --execute demo.ipynb   # this notebook
```"""),
]


def build() -> None:
    nb = nbf.v4.new_notebook()
    for kind, src in CELLS:
        cell = nbf.v4.new_markdown_cell(src) if kind == "md" else nbf.v4.new_code_cell(src)
        nb.cells.append(cell)
    nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
    nb.metadata["language_info"] = {"name": "python"}
    nbf.write(nb, str(OUT))
    print(f"wrote {OUT} ({len(nb.cells)} cells)")


if __name__ == "__main__":
    build()
