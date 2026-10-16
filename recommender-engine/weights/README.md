# published checkpoints

two small cpu checkpoints (pytorch). load them with the engine's builders, not raw `torch.load`.

| file | role | variant | params | val ndcg@10 | test ndcg@10 |
|---|---|---|---|---|---|
| `wide_rank_offline.pt` | **the offline ranker** (this repo's model) | wide | 122,773 | 0.5143 | 0.5332 |
| `taste_sim.pt` | sim-native ranker that drives the policy ladder | wide | 122,773 | — | — |

`wide_rank_offline.pt` is trained on real logs with the masked-BCE + LambdaRank objective for ~16
epochs (best epoch 11, early-stopped). `taste_sim.pt` is the same architecture trained on simulator
logs (`run_learn.py --variant wide --loss hybrid`) and is what the closed-loop simulator scores with.

each file is a dict: `state_dict`, `variant`, `model_cfg`, `feature_list` (for the offline net) and the
full per-epoch `history`.

```python
import torch, yaml
from feed.model import build_model, load_meta

policy = yaml.safe_load(open("config/policy.yaml"))
ckpt = torch.load("weights/wide_rank_offline.pt", map_location="cpu", weights_only=False)
model = build_model(load_meta(), policy, variant=ckpt["variant"])
model.load_state_dict(ckpt["state_dict"]); model.eval()
```

the simulator auto-detects the variant from the checkpoint, so point it straight at this folder:

```python
policy["_models_dir"] = "weights"; policy["_taste_ckpt"] = "taste_sim.pt"
```

reproduce: `run_train.py --variant wide --loss lambdarank` (offline net) and
`run_learn.py --variant wide --loss hybrid` (sim ranker).
