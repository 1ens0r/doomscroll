"""Intuition: The hidden content space makes item appeal *user-conditional*. A bot's preference has
two parts: a category block that is its real decay-share taste (so what it likes is learnable from
behaviour) and a small latent block for within-category taste (so two users in the same category can
want different items). Item appeal is the bot's category affinity plus its latent item affinity, so
the *same* item is worth more to some users than others and global popularity cannot rank for one.
"""
from __future__ import annotations

import numpy as np


def _unit(x: np.ndarray) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-9)


def item_idiosyncrasy(n_items: int, dim: int, rng: np.random.Generator) -> np.ndarray:
    """Per-item latent direction in a small space, shared across users (within-category taste)."""
    return _unit(rng.normal(size=(n_items, dim))).astype(np.float32)


def preference_blocks(taste: np.ndarray, idio_dim: int, rng: np.random.Generator
                      ) -> tuple[np.ndarray, np.ndarray]:
    """Category preference = standardised real taste; latent preference = a random unit direction.

    Standardising by the population spread makes the category block comparable across bots; `taste`
    is the row-normalised decay share that already ranks each bot's preferred categories.
    """
    ref = 1.0 / taste.shape[1]
    std = max(float(taste.std()), 1e-6)
    pref_cat = np.clip((taste - ref) / std, -6.0, 6.0)
    pref_idio = _unit(rng.normal(size=(len(taste), idio_dim)))
    return pref_cat.astype(np.float32), pref_idio.astype(np.float32)
