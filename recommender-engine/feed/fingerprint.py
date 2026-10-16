"""Intuition: A dataset fingerprint so two runs can be proven to have seen the same rows: a
content hash per split plus the hashes of the preprocessing and scaler configs that shaped them."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from feed.data import ART, load_split

HASH_COLS = ["user_id", "item_id", "ts", "tier", "is_search"]


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def split_fingerprint(split: str) -> dict:
    """Row count, time span, and a SHA-256 over the id/label columns of one split."""
    df = load_split(split, columns=HASH_COLS)
    h = hashlib.sha256()
    for col in HASH_COLS:
        h.update(np.ascontiguousarray(df[col].to_numpy()).tobytes())
    return {
        "n_rows": int(len(df)),
        "n_users": int(df["user_id"].nunique()),
        "n_items": int(df["item_id"].nunique()),
        "ts_min": int(df["ts"].min()),
        "ts_max": int(df["ts"].max()),
        "sha256": h.hexdigest(),
    }


def file_hash(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def dataset_fingerprint(splits: tuple[str, ...] = ("train", "val", "test")) -> dict:
    """Fingerprint all splits plus the configs that define the task."""
    cfg_path = ART / "preprocess_config.json"
    out: dict = {
        "config": {"preprocess_config": file_hash(cfg_path)},
        "splits": {s: split_fingerprint(s) for s in splits},
    }
    scaler_path = ART / "scaler.json"
    if scaler_path.exists():
        out["config"]["scaler"] = file_hash(scaler_path)
    out["config"]["tier_gains"] = json.loads(cfg_path.read_text())["tier_gains"]
    out["config"]["split_cuts"] = json.loads(cfg_path.read_text())["split_cuts"]
    return out


if __name__ == "__main__":
    print(json.dumps(dataset_fingerprint(), indent=1))
