"""Small mathematical and CASTER-tree helpers used by the one-shot core.

The former module also contained the retired manifest/branch-TSV reader. Keeping
that reader here made it easy for new code to accidentally rebuild the old cache
chain, so the maintained surface is limited to helpers still used by the
one-shot observation and tree-bank layers.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from statistics import mean

from .caster import caster_raw_s_cu_length, caster_raw_s_scores


@dataclass(frozen=True)
class CuLengthStats:
    n_internal_branches: int
    n_raw_s_branches: int
    n_structural_root_branches: int
    n_infinite_replaced: int
    min_length: float
    max_length: float
    mean_length: float


def parse_float(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def import_dendropy():
    try:
        import dendropy
        from dendropy.model import coalescent
    except ImportError as exc:
        raise RuntimeError(
            "DendroPy is required for gene-tree simulation and pruning. "
            "Run this script in the iILS conda environment."
        ) from exc
    return dendropy, coalescent


def relabel_tree_edges_from_caster_raw_scores(
    containing_tree,
    *,
    max_cu_length: float,
) -> CuLengthStats:
    lengths: list[float] = []
    n_internal = 0
    n_raw_s = 0
    n_structural_root = 0
    n_infinite = 0
    root = containing_tree.seed_node

    for node in containing_tree.postorder_node_iter():
        if node.parent_node is None:
            continue
        if node.is_leaf():
            if node.edge.length is None:
                node.edge.length = 0.0
            continue
        n_internal += 1
        scores = caster_raw_s_scores(node.label)
        if scores is None:
            if node.label is None and node.edge.length is None and node.parent_node is root:
                node.edge.length = 0.0
                n_structural_root += 1
                continue
            continue
        n_raw_s += 1
        length, was_infinite = caster_raw_s_cu_length(
            scores,
            infinite_length=max_cu_length,
        )
        if was_infinite:
            n_infinite += 1
        node.edge.length = length
        lengths.append(length)

    n_expected_raw_s = n_internal - n_structural_root
    if n_raw_s != n_expected_raw_s:
        raise ValueError(
            "CASTER support tree is missing raw s1/s2/s3 scores for "
            f"{n_expected_raw_s - n_raw_s} internal branch(es) "
            f"(raw_s={n_raw_s}, structural_root={n_structural_root}, "
            f"internal_nonleaf_nonroot={n_internal})"
        )
    if not lengths:
        raise ValueError("CASTER support tree has no internal branches to relabel")
    return CuLengthStats(
        n_internal_branches=n_internal,
        n_raw_s_branches=n_raw_s,
        n_structural_root_branches=n_structural_root,
        n_infinite_replaced=n_infinite,
        min_length=min(lengths),
        max_length=max(lengths),
        mean_length=mean(lengths),
    )


def species_tree_cu_text(containing_tree) -> str:
    return containing_tree.as_string(
        schema="newick",
        suppress_annotations=True,
        suppress_internal_node_labels=True,
        suppress_rooting=True,
        unquoted_underscores=True,
    ).strip() + "\n"


def load_cu_species_tree(caster_tree: Path, max_cu_length: float):
    dendropy, coalescent = import_dendropy()
    taxa = dendropy.TaxonNamespace()
    containing_tree = dendropy.Tree.get(
        path=str(caster_tree),
        schema="newick",
        taxon_namespace=taxa,
        preserve_underscores=True,
        suppress_internal_node_taxa=True,
    )
    containing_tree.is_rooted = True
    stats = relabel_tree_edges_from_caster_raw_scores(
        containing_tree,
        max_cu_length=max_cu_length,
    )
    return dendropy, coalescent, taxa, containing_tree, species_tree_cu_text(containing_tree), stats


def species_penalty(row: dict[str, str], epsilon: float = 1e-12) -> float:
    values = [
        parse_float(row.get("species_p1", "")),
        parse_float(row.get("species_p2", "")),
        parse_float(row.get("species_p3", "")),
    ]
    if any(not math.isfinite(value) for value in values):
        return math.nan
    values.sort(reverse=True)
    ratio = 2.0 * values[1] + values[2]
    return -math.log(max(epsilon, ratio))
