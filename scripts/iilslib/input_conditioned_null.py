from __future__ import annotations

import hashlib
import json

import numpy as np


def policy_sample_indices(
    eligible_indices: list[int],
    sample_size: int,
    seed: int,
) -> list[int]:
    """Choose a reproducible without-replacement policy sample."""
    if sample_size < 0:
        raise ValueError("sample_size must be nonnegative")
    if seed < 0:
        raise ValueError("seed must be nonnegative")
    if sample_size == 0 or sample_size >= len(eligible_indices):
        return list(eligible_indices)
    rng = np.random.default_rng(seed)
    return sorted(
        int(index)
        for index in rng.choice(
            np.asarray(eligible_indices, dtype=np.int64),
            size=sample_size,
            replace=False,
        )
    )


def sample_gene_tree_panel(
    *,
    locus_index: int,
    panel_seed_base: int,
    reservoir_size: int,
    panel_size: int,
) -> np.ndarray:
    """Return a deterministic sorted panel of unique global gene-tree IDs.

    The stream is deliberately implemented independently of NumPy's RNG so that
    panel membership is stable across NumPy versions and worker processes.
    """
    if reservoir_size <= 0:
        raise ValueError("reservoir_size must be positive")
    if panel_size <= 0:
        raise ValueError("panel_size must be positive")
    if panel_size > reservoir_size:
        raise ValueError("panel_size cannot exceed reservoir_size")
    if panel_seed_base < 0:
        raise ValueError("panel_seed_base must be nonnegative")
    if locus_index < 0:
        raise ValueError("locus_index must be nonnegative")
    seed_object = {
        "schema": "iils-panel-seed-v2",
        "locus_index": locus_index,
        "panel_seed_base": panel_seed_base,
    }
    seed_bytes = json.dumps(
        seed_object,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    state = int.from_bytes(hashlib.sha256(seed_bytes).digest()[:8], "big")
    mask = (1 << 64) - 1
    limit = ((1 << 64) // reservoir_size) * reservoir_size
    selected: set[int] = set()
    while len(selected) < panel_size:
        state = (state + 0x9E3779B97F4A7C15) & mask
        value = state
        value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & mask
        value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & mask
        value ^= value >> 31
        if value >= limit:
            continue
        selected.add(int(value % reservoir_size))
    return np.asarray(sorted(selected), dtype=np.int64)
