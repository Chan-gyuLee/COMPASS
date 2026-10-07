"""Explicit embedding selection, including dropout-ensemble embeddings."""

import hashlib
from pathlib import Path
import tempfile

import numpy as np


def load_context(sequence, cache_dir, single_embeddings=None, pair_embeddings=None):
    from bioemu.sample import get_context_chemgraph

    cache_dir = Path(cache_dir).expanduser()
    cache_dir.mkdir(parents=True, exist_ok=True)
    if (single_embeddings is None) != (pair_embeddings is None):
        raise ValueError("Provide both single and pair embedding files, or neither")
    if single_embeddings is None:
        return get_context_chemgraph(
            sequence=sequence, cache_embeds_dir=str(cache_dir / "embeddings")
        )
    single = np.load(single_embeddings, allow_pickle=False)
    pair = np.load(pair_embeddings, allow_pickle=False)
    length = len(sequence)
    if single.ndim != 2 or single.shape[0] != length or single.shape[1] < 1:
        raise ValueError("Single embeddings must have shape [L, C_single]")
    if pair.ndim == 2 and pair.shape[0] == length * length:
        pair = pair.reshape(length, length, -1)
    if pair.ndim != 3 or pair.shape[:2] != (length, length) or pair.shape[2] < 1:
        raise ValueError(
            "Pair embeddings must have shape [L, L, C_pair] or [L*L, C_pair]"
        )
    if any(
        not np.issubdtype(a.dtype, np.floating) or not np.isfinite(a).all()
        for a in (single, pair)
    ):
        raise ValueError("Embedding arrays must contain finite floating-point values")
    digest = hashlib.sha256(sequence.encode()).hexdigest()
    # A dedicated cache makes this draw explicit and leaves shared caches intact.
    with tempfile.TemporaryDirectory(prefix="compass-embeds-", dir=cache_dir) as stage:
        np.save(Path(stage) / f"{digest}_single.npy", single.astype(np.float32))
        np.save(Path(stage) / f"{digest}_pair.npy", pair.astype(np.float32))
        return get_context_chemgraph(sequence=sequence, cache_embeds_dir=stage)
