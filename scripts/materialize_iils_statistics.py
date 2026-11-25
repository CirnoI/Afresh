#!/usr/bin/env python3
"""Materialize support-weighted iILS scores from raw statistics TSVs."""

from __future__ import annotations

import argparse
from pathlib import Path

from iilslib.statistics import materialize_gene_scores, materialize_msa_scores


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-type", choices=("gene-tree", "msa-list"), required=True)
    parser.add_argument("--statistics", type=Path, required=True)
    parser.add_argument("--item-statistics", type=Path)
    parser.add_argument("--support-tree", type=Path, required=True)
    parser.add_argument("--branch-output", type=Path)
    parser.add_argument("--score-output", type=Path, required=True)
    parser.add_argument(
        "--score-transform",
        choices=("mass", "raw-alt", "gap", "winner"),
        default="mass",
    )
    parser.add_argument("--n-gene-trees", type=int)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.input_type == "msa-list":
        if args.item_statistics is None or args.branch_output is None:
            raise ValueError(
                "msa-list input requires --item-statistics and --branch-output"
            )
        if args.n_gene_trees is not None:
            raise ValueError("--n-gene-trees is only valid for gene-tree input")
        materialize_msa_scores(
            args.statistics,
            args.item_statistics,
            args.support_tree,
            args.branch_output,
            args.score_output,
            score_transform=args.score_transform,
        )
    else:
        if args.n_gene_trees is None:
            raise ValueError("gene-tree input requires --n-gene-trees")
        if args.item_statistics is not None or args.branch_output is not None:
            raise ValueError(
                "--item-statistics and --branch-output are only valid for msa-list input"
            )
        materialize_gene_scores(
            args.statistics,
            args.support_tree,
            args.score_output,
            n_gene_trees=args.n_gene_trees,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
