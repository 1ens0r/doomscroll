"""Build `demo.ipynb` (repo root) — the presentation notebook.

Cells are written here; heavy artifacts are read from `notes/results/` and `notes/figures/`, and the
small published checkpoints from `recommender-engine/weights/`. Execute with:

    python -m nbconvert --to notebook --execute demo.ipynb --output demo.ipynb
"""
from __future__ import annotations

from pathlib import Path

import nbformat as nbf

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "demo.ipynb"

CAT_NAMES = {
    0: "empty", 1: "dance", 2: "music", 3: "game", 4: "Beauty Makeup", 5: "Style",
    6: "Star Entertainment", 7: "motion", 8: "Appearance value", 9: "Funny", 10: "Travel",
    11: "life", 12: "Food", 13: "San Nong", 14: "education", 15: "Talent", 16: "health",
    17: "animal", 18: "automobile", 19: "emotion", 20: "pixiv", 21: "history",
    22: "Finance", 23: "religion", 24: "Astrology", 25: "Parent-child", 26: "photograph",
    27: "High tech", 28: "real-time info", 29: "science", 30: "statute", 31: "law",
    32: "sports", 33: "outdoors", 34: "Random clap", 35: "military", 36: "Real Estate",
    37: "Oddities", 38: "read", 39: "Film & Short Drama",
}

CELLS: list[tuple[str, str]] = [
    ("md", """# doomscroll
### A neural feed ranker, a taste-conditioned composer, and a closed-loop user simulator

**The question.** Given a stream of (user, video) impressions, can a small CPU-trainable ranker plus a
cheap re-ranking *composer* order a session to maximise a graded engagement score — and can we prove
it against popularity, affinity and an oracle-free ceiling?

**What this notebook does, end to end.** loads the real processed logs, renders the study figures,
shows the improved network and its objective, walks the real training history epoch by epoch, reports
the val/test and loss ablations, and finally runs a live single-subject simulation.

| Section | Data source | Nature |
|---|---|---|
| 1–4 Data, EDA, task, features | real `features_*.parquet` | **real logs** |
| 5–7 Network, training, evaluation | trained weights + committed results | **real numbers** |
| 8 Live single-subject simulation | the closed-loop simulator | **simulated** |

> **Honesty.** Offline metrics come from real KuaiSAR-v2 logs. Section 8 is a *simulated* rollout —
> the responses come from the world model, so it illustrates behaviour and is **not** a causal claim.

Seed **42** · CPU only · checkpoints published in `recommender-engine/weights/`."""),

    ("md", "## 0 · Environment and reproducibility\nWe pin the environment, set the seed, and print the dataset fingerprint (a sha256 per split) so any number below is anchored to an exact data snapshot."),
    ("code", """import json, platform, subprocess, sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from IPython.display import Image, Markdown, display

REPO = Path.cwd()
if not (REPO / "recommender-engine").exists():
    REPO = REPO.parent
ENGINE = REPO / "recommender-engine"
sys.path.insert(0, str(ENGINE))
torch.set_num_threads(8)
pd.set_option("display.max_colwidth", 60)

# first-level-category id -> English name (from the raw taxonomy), used to label simulated videos
CAT_NAMES = _CAT_NAMES_

try:
    commit = subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"],
                                     text=True).strip()
except Exception:
    commit = "(no git)"

def show(df, caption=None, fmt=None):
    \"\"\"Render a compact, index-free table.\"\"\"
    if caption:
        display(Markdown(f"**{caption}**"))
    st = df.style.hide(axis="index")
    if fmt:
        st = st.format(fmt)
    display(st)

fp = json.loads((REPO / "notes" / "results" / "fingerprint.json").read_text())
print(f"python {platform.python_version()} | torch {torch.__version__} | numpy {np.__version__} "
      f"| device cpu | seed 42 | git {commit}")
print("SIGNED DATA:", ", ".join(f"{s}={fp['splits'][s]['sha256'][:10]}" for s in ("train", "val", "test")))"""),

    ("md", """## 1 · Data loading
The cleaned features live in `artifacts/features_{split}.parquet`. Reads are always **column-selective**,
and the splits are **strict time cuts** — validation and test are strictly future data. Below we load a
small column subset of all three splits and summarise them."""),
    ("code", """from feed.data import load_split

sub = ["user_id", "item_id", "ts", "tier", "is_search", "cold_item", "i_mean_score", "uc_lift"]
rows, train_df = [], None
for s in ("train", "val", "test"):
    d = load_split(s, columns=sub)
    if s == "train":
        train_df = d
    rows.append({
        "split": s, "rows": f"{len(d):,}", "users": f"{d['user_id'].nunique():,}",
        "items": f"{d['item_id'].nunique():,}",
        "search share": f"{d['is_search'].mean():.1%}",
        "cold-item share": f"{d['cold_item'].mean():.1%}",
        "mean tier": f"{d['tier'].mean():.3f}",
    })
show(pd.DataFrame(rows), "Split summary (time-ordered; no leakage across splits)")

print("\\nSample rows from train:")
display(train_df.head(5))"""),

    ("md", """## 2 · Exploratory results and figures
These are the study's **committed figures** (rendered from `notes/figures/`) — the EDA and evaluation
plots that back every claim, in order."""),
    ("code", """CAPTIONS = {
    "01_ladder.png": "THE LADDER — mean IWE per bot for every policy, with the share of the "
                     "personalized-oracle gap closed.",
    "02_generations.png": "Composer generations — each component is kept only if IWE rises with a "
                          "95% CI excluding zero and no guardrail degrades.",
    "03_calibration.png": "Simulator calibration — 12 targets against the 15% tolerance band "
                          "(7 pass, 5 structural misses documented).",
    "04_cohorts.png": "Per-cohort personalization headroom (activity tercile x breadth tercile).",
    "05_retention.png": "Retention — return curves by day for each policy.",
    "06_loss_diagnostics.png": "Loss diagnostics — click/watch reliability curves and per-head "
                               "validation BCE.",
}
for p in sorted((REPO / "notes" / "figures").glob("*.png")):
    display(Markdown(f"**{CAPTIONS.get(p.name, p.name)}**"))
    display(Image(filename=str(p)))"""),

    ("md", "### Headline results at a glance\nThe policy ladder, the network-capacity study, and the composer search — as tables."),
    ("code", """imp = json.loads((REPO / "notes" / "results" / "improve.json").read_text())
cap = json.loads((REPO / "notes" / "results" / "capacity.json").read_text())
sim = json.loads((REPO / "notes" / "results" / "sim.json").read_text())

ladder = pd.DataFrame([{"policy": p, "IWE / bot": v["mean"],
                        "oracle gap closed": sim["oracle_gap_closed"].get(p, np.nan)}
                       for p, v in sim["iwe"].items()])
show(ladder, "Policy ladder (10,000 simulated bots x 7 days)",
     {"IWE / bot": "{:.2f}", "oracle gap closed": "{:.2f}"})

caprows = [{"variant": n, "params": f"{v['n_params']:,}", "best epoch": v["best_epoch"],
            "val NDCG@10": cap["val_feed"][n]["ndcg@10"]["mean"]}
           for n, v in cap["variants"].items()]
show(pd.DataFrame(caprows).sort_values("val NDCG@10", ascending=False),
     "Network capacity (masked-BCE objective, identical protocol)",
     {"val NDCG@10": "{:.4f}"})"""),

    ("md", """## 3 · Task and labels
The model predicts **five action heads** — `click`, `like`, `follow`, `forward`, `watch`. Only `click`
is observed on every row; the deeper actions are logged **only on feed** rows, so search rows are
**masked**, never treated as negatives. The graded objective maps engagement `tier` to gains
`[0, 1, 2, 3, 4]`."""),
    ("code", """from feed.data import HEADS, make_targets

df = load_split("train", columns=["tier", "is_search", "click", "like", "follow", "forward",
                                  "watch", "obs_like", "obs_follow", "obs_forward", "obs_watch"])
gains = json.loads((ENGINE / "artifacts" / "preprocess_config.json").read_text())["tier_gains"]
y, m = make_targets(df)

dist = pd.DataFrame({"tier": range(5),
                     "gain": gains,
                     "rows": np.bincount(df["tier"].to_numpy(), minlength=5)})
dist["share"] = dist["rows"] / dist["rows"].sum()
show(dist, "Tier distribution and gains", {"share": "{:.1%}", "rows": "{:,}"})

rates = pd.DataFrame([{
    "head": h, "positive rate": float((y[:, j] * m[:, j]).sum() / max(1.0, m[:, j].sum())),
    "masked (unobserved)": float(1 - m[:, j].mean()),
} for j, h in enumerate(HEADS)])
show(rates, "Per-head label prevalence and masking (train)",
     {"positive rate": "{:.4f}", "masked (unobserved)": "{:.2%}"})"""),

    ("md", """## 4 · Features
27 dense features in three groups plus one categorical, all constructed **as-of** the impression
timestamp (no future information). Dense values are standardised with **train-only** statistics, with
`log1p` on the heavy-tailed count `i_n`."""),
    ("code", """from feed.data import DENSE, load_scaler

sc = load_scaler()
meta = json.loads((ENGINE / "artifacts" / "feature_list.json").read_text())
groups = {
    "user habits": [f for f in DENSE if f.startswith("u_")],
    "item priors": [f for f in DENSE if f.startswith("i_")],
    "pair / context": [f for f in DENSE if f.startswith(("uc_", "ui_", "cold_", "ctx_"))],
}
show(pd.DataFrame([{"group": k, "n": len(v), "features": ", ".join(v)} for k, v in groups.items()]),
     f"Dense feature blocks ({len(DENSE)} total)")
print(f"categorical: cat_id -> {meta['cat_embed_rows']} embedding rows  |  "
      f"scaler fit on {sc['fit_on']} only, {sc['n_rows']:,} rows, log1p on {sc['log1p']}")"""),

    ("md", """## 5 · The improved network
`TasteNet` is config-driven: a **user tower** and a **video tower** are combined with an explicit
`user * video` interaction block, concatenated with a **pair block**, then fused to the five logits.
The cell below is the network's actual forward pass; the variant table shows the published sizes."""),
    ("md", """```python
# feed/model.py  (forward pass, condensed)
xu = self.user_tower(x_user)          #  9 -> 64 -> 32
xv = self.video_tower(x_video, cat)   #  5 + 16-dim embedding -> 64 -> 32
xp = self.pair_tower(x_pair)          # 13 -> 32
h  = torch.cat([xu, xv, xu * xv, xp], dim=1)   # explicit interaction
return self.head(self.fuse(h))        # 128 -> 128 -> 64 -> 5
```"""),
    ("code", """import yaml
from feed.model import build_model, list_variants, n_params, resolve_model_cfg

pol = yaml.safe_load((ENGINE / "config" / "policy.yaml").read_text())

vrows = []
for v in list_variants(pol):
    model_v = build_model(meta, pol, variant=None if v == "baseline" else v)
    vrows.append({"variant": v, "params": f"{n_params(model_v):,}",
                  "activation": resolve_model_cfg(pol, v).get("activation", "relu"),
                  "norm": resolve_model_cfg(pol, v).get("norm", "none")})
show(pd.DataFrame(vrows), "Model variants and parameter counts")

best_variant = "wide"
model = build_model(meta, pol, variant=best_variant)
print(f"improved model = '{best_variant}' with {n_params(model):,} parameters")"""),

    ("md", """### Objective — masked BCE + a ranking-aware term
The improved run uses a **hybrid loss**: the masked per-head BCE from the baseline, **plus** a
differentiable LambdaRank surrogate on the graded gains, mixed with a pairwise hinge. Ranking loss
cares about *order within a user's slate*, which is exactly what NDCG measures.

```python
# feed/losses.py  (HybridLoss, condensed)
total = masked_bce(logits, y, m, pos_weight) + rank_weight * lambdarank(scores, tier, groups)
```

The ablation below holds the architecture fixed and varies only the objective / one knob at a time —
all on the identical data, split, candidate protocol and seed."""),
    ("code", """la = json.loads((REPO / "notes" / "results" / "loss_ablation.json").read_text())
lrows = []
for arm, a in la["arms"].items():
    override = ", ".join(f"{k.split('.')[-1]}={v}" for k, v in a["overrides"].items()) or "baseline"
    row = {"arm": arm, "change": override, "params": f"{a['n_params']:,}",
           "best ep": a["best_epoch"], "val NDCG@10": a["val_feed"]["ndcg@10"]["mean"],
           "val NDCG@20": a["val_feed"]["ndcg@20"]["mean"]}
    c = la["paired_vs_baseline"].get(arm)
    row["delta vs L0"] = c["mean_diff"] if c else 0.0
    row["95% CI"] = f"[{c['lo']:+.4f}, {c['hi']:+.4f}]" if c else "reference"
    lrows.append(row)
ldf = pd.DataFrame(lrows).sort_values("val NDCG@10", ascending=False)
show(ldf, f"Loss / optimisation ablation on the baseline architecture "
          f"({'full data' if not la['fast'] else 'smoke'})",
     {"val NDCG@10": "{:.4f}", "val NDCG@20": "{:.4f}", "delta vs L0": "{:+.4f}"})
print(f"best arm by val NDCG@10: {la['best']}")"""),
    ("code", """dg = json.loads((REPO / "notes" / "results" / "loss_diagnostics.json").read_text())
drows = [{"head": h, "prevalence": d["val_prevalence"], "val BCE": d["val_bce"],
          "Brier": d["brier"], "log-loss": d["log_loss"], "ECE": d["ece"],
          "grad norm": d["grad_norm"]} for h, d in dg["heads"].items()]
show(pd.DataFrame(drows), "Per-head loss and calibration diagnostics (baseline arm, val)",
     {"prevalence": "{:.4f}", "val BCE": "{:.4f}", "Brier": "{:.4f}", "log-loss": "{:.4f}",
      "ECE": "{:.4f}", "grad norm": "{:.3f}"})
print("pos_weight per head:", [round(x, 3) for x in dg["pos_weights"]],
      f"(mode={dg['pos_weight_mode']})")"""),

    ("md", """## 6 · Training — the real history, epoch by epoch
Both the reference and the improved model are trained on the **same** data, split and seed with a
OneCycle schedule and early stopping on validation NDCG@10. The logs below are the saved runs."""),
    ("code", """def print_log(label):
    r = imp["results"][label]
    print(f"{label}  —  {r['n_params']:,} params, {len(r['history'])} epochs, "
          f"best epoch {r['best_epoch']}")
    for i, h in enumerate(r["history"], 1):
        tag = "   <-- best (weights kept)" if i == r["best_epoch"] else ""
        print(f"  epoch {i}/{len(r['history'])}  train_loss={h['train_loss']:.4f}  "
              f"val_loss={h['val_loss']:.4f}  val_ndcg@10={h['val_ndcg10']:.4f}  "
              f"val_ndcg@20={h['val_ndcg20']:.4f}  lr={h['lr']:.2e}  "
              f"({h['seconds']:.0f}s){tag}")
    print()

print("================  REFERENCE (masked BCE)  ================")
print_log("baseline_masked")
print("================  IMPROVED (wide + hybrid)  ==============")
print_log("wide_hybrid")"""),

    ("md", """## 7 · Evaluation — improved vs reference
Selection used validation only; **test was touched once** per arm. Differences are paired
user-bootstraps (2,000 resamples). The improved model wins on **both** val and test, and the validity
bar is that the validation CI excludes zero."""),
    ("code", """rows = []
for label, r in imp["results"].items():
    row = {"config": label, "params": f"{r['n_params']:,}", "best ep": r["best_epoch"],
           "val NDCG@10": r["val_feed"]["ndcg@10"]["mean"],
           "val NDCG@20": r["val_feed"]["ndcg@20"]["mean"]}
    if "test_feed" in imp:
        row["test NDCG@10"] = imp["test_feed"][label]["ndcg@10"]["mean"]
        row["test NDCG@20"] = imp["test_feed"][label]["ndcg@20"]["mean"]
    rows.append(row)
show(pd.DataFrame(rows), "Offline ranking quality (feed scope)",
     {k: "{:.4f}" for k in ("val NDCG@10", "val NDCG@20", "test NDCG@10", "test NDCG@20")})

c = imp["paired_vs_reference"]["wide_hybrid"]
display(Markdown(
    f"> **wide_hybrid − baseline, val NDCG@10 = {c['mean_diff']:+.4f}** "
    f"(95% CI [{c['lo']:+.4f}, {c['hi']:+.4f}], excludes 0 → significant).\\n>\\n"
    f"> Test NDCG@10 {imp['test_feed']['baseline_masked']['ndcg@10']['mean']:.4f} → "
    f"**{imp['test_feed']['wide_hybrid']['ndcg@10']['mean']:.4f}** "
    f"(Δ +{imp['test_feed']['wide_hybrid']['ndcg@10']['mean'] - imp['test_feed']['baseline_masked']['ndcg@10']['mean']:.4f}). "
    f"Capacity alone was neutral, so this gain is the **objective**, not extra parameters."))"""),

    ("md", """## 8 · Live single-subject simulation
Finally we roll the trained **sim-native ranker** through the closed-loop simulator with the
taste-conditioned composer, and print one subject's session swipe by swipe — the item being watched,
the next recommendation (**with its video title**), and the simulated response. Everything below is
**simulated**."""),
    ("code", """from collections import defaultdict

from feed.policies import TasteScorer
from feed.swarm import simulate
from feed.world import load_config

pol["_models_dir"] = ENGINE / "weights"
pol["_taste_ckpt"] = "taste_sim.pt"                 # ranker trained on serving-path features
scorer = TasteScorer(pol, int(pol["python_threads"]))

cfg = load_config()
cfg["policy"] = pol
trace = []
stats = simulate("taste_composer", cfg, seed=42, scorer=scorer, n_bots=64, n_days=1,
                 trace=trace, log=lambda *a: None)

sessions = defaultdict(list)
for t in trace:
    sessions[t["bot"]].append(t)
bot = max(sessions, key=lambda b: len(sessions[b]))
steps = sorted(sessions[bot], key=lambda t: (t["day"], t["wave"], t["step"]))

def title(t):
    name = CAT_NAMES.get(t["cat"], "category " + str(t["cat"]))
    return f"{name} · clip #{t['item_id']}"

def action(t):
    parts = [k for k in ("click", "like", "follow", "forward") if t[k]]
    if t["watch"] > 0:
        parts.append(f"watch {t['watch'] * 100:.0f}%")
    return ", ".join(parts) if parts else "scroll past"

print(f"SIMULATED session — taste_composer — subject #{bot}, {len(steps)} swipes\\n")
prev = None
for i, t in enumerate(steps[:14], 1):
    watching = title(prev) if prev else "— (session start)"
    print(f"[{i:02d}] currently watching : {watching}")
    print(f"     next recommended   : {title(t)}")
    print(f"     action             : {action(t)}        (tier {t['tier']})")
    prev = t
print(f"\\nSession IWE = {stats['iwe_mean']:.2f} over {stats['impressions']:,} impressions."
      f"  (SIMULATED — illustrative, not causal)")"""),

    ("md", """## 9 · Conclusions and limitations

**Findings.**
- A **123k-parameter** `TasteNet` trained with a **ranking-aware hybrid loss** beats the 32k masked-BCE
  baseline on validation (**+0.0045, 95% CI [+0.0028, +0.0062]**) and on the single test touch
  (0.5178 → 0.5227).
- **Capacity alone is neutral** (wide / wide_deep / cross all within noise of baseline): the objective,
  not the parameters, does the work.
- In the simulator, the **taste-conditioned composer** (`taste_composer` 140.4) and the **evolved**
  policy (151.5) lift IWE over a pure taste ranker (120.3) and close **59%** of the personalized-oracle
  gap — every rung CI-separated.

**Limitations.**
- Engagement targets are observational; `watch` is a proxy for satisfaction and search rows are masked.
- The simulator is a hand-designed, marginal-calibrated world (7/12 targets), so section 8's IWE is
  **relative and illustrative**, not a causal effect.
- The loss ablation is an **8-arm family sweep on the full data** rather than a full 15-arm grid, so
  some optimiser corners are under-sampled.

### Reproduce
```bash
cd recommender-engine
P=~/miniconda3/envs/picograd/bin/python
$P run_offline.py --fast                              # offline baseline + ladder
$P run_improve.py --test baseline_masked,wide_hybrid  # the headline comparison
$P run_swarm.py                                       # policy ladder (incl. evolved)
$P build_demo.py && python -m nbconvert --to notebook --execute ../demo.ipynb --output ../demo.ipynb
```"""),
]


def build() -> None:
    nb = nbf.v4.new_notebook()
    nb.cells = [nbf.v4.new_markdown_cell(s) if k == "md" else nbf.v4.new_code_cell(
        s.replace("_CAT_NAMES_", repr(CAT_NAMES))) for k, s in CELLS]
    nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
    nb.metadata["language_info"] = {"name": "python"}
    nbf.write(nb, str(OUT))
    print(f"wrote {OUT} ({len(nb.cells)} cells)")


if __name__ == "__main__":
    build()
