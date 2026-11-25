#!/usr/bin/env python3
"""Filter gene trees by a materialized iILS loss and write a Newick file."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Remove NaN-loss gene trees and the highest-loss fraction."
    )
    parser.add_argument("--scores", required=True, help="iils_scores.tsv")
    parser.add_argument("--gene-trees", required=True, help="Original gene tree file")
    parser.add_argument("--output", required=True, help="Filtered gene tree output")
    parser.add_argument("--removed", required=True, help="TSV describing removed gene trees")
    parser.add_argument("--kept", required=True, help="TSV describing kept gene trees")
    parser.add_argument("--score-column", default="loss_norm_mean")
    parser.add_argument("--top-fraction", type=float, default=0.05)
    return parser.parse_args()


def read_parenthesized_trees(path: Path) -> list[tuple[int, str]]:
    trees: list[tuple[int, str]] = []
    for line_no, line in enumerate(path.read_text().splitlines(), start=1):
        stripped = line.strip()
        if "(" in stripped:
            trees.append((line_no, stripped))
    return trees


def read_scores(path: Path, score_column: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if score_column not in (reader.fieldnames or []):
            raise ValueError(f"{path} does not contain score column {score_column!r}")
        for row in reader:
            gene_id = int(row["gene_id"])
            try:
                score = float(row[score_column])
            except ValueError:
                score = math.nan
            rows.append({"gene_id": gene_id, "score": score})
    return rows


def main() -> None:
    args = parse_args()
    scores_path = Path(args.scores)
    gene_tree_path = Path(args.gene_trees)
    output_path = Path(args.output)
    removed_path = Path(args.removed)
    kept_path = Path(args.kept)

    if not (0 <= args.top_fraction < 1):
        raise ValueError("--top-fraction must be in [0, 1)")

    trees = read_parenthesized_trees(gene_tree_path)
    scores = read_scores(scores_path, args.score_column)
    if len(trees) != len(scores):
        raise ValueError(
            f"Effective tree count mismatch: {len(trees)} parenthesized trees in "
            f"{gene_tree_path}, but {len(scores)} rows in {scores_path}"
        )

    finite = [row for row in scores if math.isfinite(float(row["score"]))]
    n_top = math.ceil(len(finite) * args.top_fraction)
    top_gene_ids = {
        int(row["gene_id"])
        for row in sorted(finite, key=lambda r: float(r["score"]), reverse=True)[:n_top]
    }
    nan_gene_ids = {int(row["gene_id"]) for row in scores if not math.isfinite(float(row["score"]))}
    remove_ids = top_gene_ids | nan_gene_ids

    output_path.parent.mkdir(parents=True, exist_ok=True)
    removed_path.parent.mkdir(parents=True, exist_ok=True)
    kept_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w") as out, removed_path.open("w", newline="") as rem, kept_path.open("w", newline="") as kep:
        fieldnames = ["gene_id", "input_line", args.score_column, "reason"]
        rem_writer = csv.DictWriter(rem, fieldnames=fieldnames, delimiter="\t")
        kep_writer = csv.DictWriter(kep, fieldnames=fieldnames, delimiter="\t")
        rem_writer.writeheader()
        kep_writer.writeheader()

        for row, (line_no, tree) in zip(scores, trees):
            gene_id = int(row["gene_id"])
            score = float(row["score"])
            if gene_id in nan_gene_ids:
                reason = "nan"
            elif gene_id in top_gene_ids:
                reason = "top_loss"
            else:
                reason = "kept"

            out_row = {
                "gene_id": gene_id,
                "input_line": line_no,
                args.score_column: "nan" if not math.isfinite(score) else f"{score:.12g}",
                "reason": reason,
            }
            if reason == "kept":
                out.write(tree + "\n")
                kep_writer.writerow(out_row)
            else:
                rem_writer.writerow(out_row)

    print(
        f"input_effective={len(scores)} finite={len(finite)} "
        f"removed_nan={len(nan_gene_ids)} removed_top={len(top_gene_ids)} "
        f"kept={len(scores) - len(remove_ids)} output={output_path}"
    )


if __name__ == "__main__":
    main()
