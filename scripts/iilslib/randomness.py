"""Explicit seeds shared by pipeline and standalone stage entry points."""

from __future__ import annotations

import argparse


DEFAULT_SEED = 43117
SEED_RULE = "explicit-module-or-global-v1"


def add_seed_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED,
                        help=f"Global random seed. Default: {DEFAULT_SEED}.")


def effective_seed(args: argparse.Namespace, name: str, value: str | int | None = None) -> int:
    override = getattr(args, name, None)
    seed = override if override is not None else (
        value if value is not None and value != "" else getattr(args, "seed", DEFAULT_SEED)
    )
    seed = int(seed)
    if not 0 <= seed <= 2147483647:
        raise ValueError(f"{name}: seed must be in [0, 2147483647]")
    return seed


def resolve_seeds(args: argparse.Namespace, *names: str) -> None:
    effective_seed(args, "seed")
    for name in names:
        setattr(args, name, effective_seed(args, name))
