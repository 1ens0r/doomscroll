# Published checkpoints

Small, ready-to-load model weights (CPU, PyTorch). Each file is a dict with `state_dict`,
`variant`, `model_cfg`, `feature_list` (or `blocks`/`cat_rows`), and the training history.

| file | role | variant | params | notes |
|---|---|---|---|---|
| `taste_offline_baseline.pt` | offline reference (Phase 1) | baseline | 32,357 | masked BCE, val NDCG@10 0.4961 |
| `baseline_masked_offline.pt` | offline reference (Phase A protocol) | baseline | 32,357 | masked BCE, re-trained under the shared Phase-A harness |
| `wide_hybrid_offline.pt` | **improved offline ranker** | wide | 122,773 | masked BCE + LambdaRank surrogate; val 0.5006 / test 0.5227 |
| `taste_sim.pt` | **sim-native ranker** | wide | 122,773 | trained on simulator logs (`run_learn.py --variant wide --loss hybrid`), drives the policy ladder |

Load with the engine's builders instead of raw `torch.load` where possible:

```python
from feed.model import build_model, load_meta
from feed.data import load_scaler
import torch, yaml

policy = yaml.safe_load(open("config/policy.yaml"))
ckpt = torch.load("weights/wide_hybrid_offline.pt", map_location="cpu", weights_only=False)
model = build_model(load_meta(), policy, variant=ckpt["variant"])
model.load_state_dict(ckpt["state_dict"]); model.eval()
```

The simulator scores live features with the **train-only** scaler and auto-detects the architecture
variant from the checkpoint, so `feed.policies.TasteScorer` can point straight at this folder:

```python
policy["_models_dir"] = "weights"; policy["_taste_ckpt"] = "taste_sim.pt"
```

Reproduce each checkpoint:
`run_offline.py` (baseline), `run_improve.py --test baseline_masked,wide_hybrid` (wide_hybrid),
`run_learn.py --variant wide --loss hybrid` (taste_sim).
