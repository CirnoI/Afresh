from __future__ import annotations

import csv
import math
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from .caster import caster_label_attrs


STATISTICS_SCHEMA_VERSION = "iils_statistics_v1"
SUPPORT_SOURCE = "caster-annotated"
GENE_STATISTICS_FIELDS = ["gene_id", "split_key", "discordance_mass"]
MSA_STATISTICS_FIELDS = [
    "locus_id",
    "alignment_path",
    "species_branch_id",
    "split_key",
    "sequence_score1",
    "sequence_score2",
    "sequence_score3",
    "sequence_score_sum",
    "q_locus",
    "n_sites_used",
    "n_taxa_present",
]
MSA_ITEM_FIELDS = [
    "locus_id",
    "alignment_path",
    "n_taxa_present",
    "n_sites_raw",
    "n_sites_used",
    "n_internal_gene_blocks",
    "n_species_branches",
]
GENE_SCORE_FIELDS = [
    "gene_id",
    "n_valid_branches",
    "discordance_mass_sum",
    "discordance_mass_mean",
    "loss_norm_sum",
    "loss_norm_mean",
    "loss_weighted_norm_sum",
    "loss_weighted_norm_mean",
    "mean_ils_ratio",
    "mean_branch_confidence",
]
STATISTICS_PROJECT_DIR = Path("Afresh")


@dataclass(frozen=True)
class BranchSupport:
    proportions: tuple[float, float, float]
    ils_ratio: float
    confidence: float

    @property
    def penalty(self) -> float:
        return -math.log(max(1e-12, self.ils_ratio))


@dataclass
class LocusAggregate:
    metadata: dict[str, str]
    preliminary_skip: bool = False
    n_valid_branches: int = 0
    n_skipped_branches: int = 0
    loss_norm_sum: float = 0.0
    loss_weighted_norm_sum: float = 0.0
    alt_winning_branches: int = 0
    sum_ils_ratio: float = 0.0
    sum_branch_confidence: float = 0.0


@dataclass
class GeneAggregate:
    n_valid_branches: int = 0
    discordance_mass_sum: float = 0.0
    loss_norm_sum: float = 0.0
    loss_weighted_norm_sum: float = 0.0
    sum_ils_ratio: float = 0.0
    sum_branch_confidence: float = 0.0


def statistics_binary_default() -> Path:
    return STATISTICS_PROJECT_DIR / "bin" / "iils-statistics"


def statistics_build_command(aster_dir: Path) -> list[str | Path]:
    make_aster_dir = aster_dir if aster_dir.is_absolute() else Path("..") / aster_dir
    return [
        "make",
        "-C",
        STATISTICS_PROJECT_DIR,
        "iils-statistics",
        f"ASTER_DIR={make_aster_dir}",
    ]


def statistics_interface_is_compatible(binary: Path) -> bool:
    try:
        completed = subprocess.run(
            [str(binary), "--help"],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return False
    text = f"{completed.stdout}\n{completed.stderr}"
    return (
        completed.returncode == 0
        and "Export per-item quartet/site-pattern statistics" in text
        and "does not infer a tree or calculate support" in text
    )


def require_statistics_interface(binary: Path) -> None:
    if statistics_interface_is_compatible(binary):
        return
    raise RuntimeError(
        f"{binary} is not the expected statistics interface. Build it with "
        "`make -C Afresh iils-statistics`."
    )


def statistics_command(
    binary: Path,
    *,
    input_type: str,
    constraint_tree: Path,
    input_path: Path,
    output_path: Path,
    threads: int,
    item_output_path: Path | None = None,
    mapping_path: Path | None = None,
    chunk_size: int | None = None,
    seed: int | None = None,
    gene_id_offset: int | None = None,
) -> list[str | Path]:
    if input_type not in {"gene-tree", "msa-list"}:
        raise ValueError(f"unsupported statistics input type: {input_type}")
    command: list[str | Path] = [
        binary,
        "--input-type",
        input_type,
        "-c",
        constraint_tree,
        "-o",
        output_path,
        "-t",
        str(threads),
    ]
    if item_output_path is not None:
        command.extend(["--item-output", item_output_path])
    if mapping_path is not None:
        command.extend(["-a", mapping_path])
    if chunk_size is not None:
        command.extend(["--chunk", str(chunk_size)])
    if seed is not None:
        command.extend(["--seed", str(seed)])
    if gene_id_offset is not None:
        if input_type != "gene-tree":
            raise ValueError("gene_id_offset is only valid for gene-tree input")
        if gene_id_offset < 0:
            raise ValueError("gene_id_offset must be nonnegative")
        command.extend(["--gene-id-offset", str(gene_id_offset)])
    command.append(input_path)
    return command


def _canonical_split(ids: Iterable[int], n_taxa: int) -> str:
    clade = sorted(set(ids))
    selected = set(clade)
    other = [index for index in range(n_taxa) if index not in selected]
    if len(clade) > len(other) or (len(clade) == len(other) and other < clade):
        clade = other
    return ",".join(str(index) for index in clade)


def caster_score_proportions(
    scores: tuple[float, float, float],
) -> tuple[float, float, float]:
    order = sorted(range(3), key=lambda index: (-scores[index], index))
    best = scores[order[0]]
    second = scores[order[1]]
    proportions = [1.0 / 3.0] * 3
    if not best > 0.0:
        return tuple(proportions)
    if second <= 0.0:
        proportions = [0.0] * 3
        proportions[order[0]] = 1.0
        return tuple(proportions)
    concordant = best / (best + 2.0 * second)
    if concordant <= 1.0 / 3.0:
        return tuple(proportions)
    if concordant >= 1.0:
        proportions = [0.0] * 3
        proportions[order[0]] = 1.0
        return tuple(proportions)
    exp_minus_t = 1.5 - 1.5 * concordant
    if exp_minus_t <= 0.0:
        proportions = [0.0] * 3
        proportions[order[0]] = 1.0
        return tuple(proportions)
    alternative = exp_minus_t / 3.0
    proportions = [alternative] * 3
    proportions[order[0]] = 1.0 - 2.0 * alternative
    return tuple(proportions)


def _support_from_scores(scores: tuple[float, float, float]) -> BranchSupport:
    proportions = caster_score_proportions(scores)
    ordered = sorted(proportions, reverse=True)
    return BranchSupport(
        proportions=proportions,
        ils_ratio=2.0 * ordered[1] + ordered[2],
        confidence=max(0.0, ordered[0] - ordered[1]),
    )


def caster_support_by_split(path: Path) -> dict[str, BranchSupport]:
    try:
        import dendropy
    except ImportError as exc:  # pragma: no cover - environment diagnostic
        raise RuntimeError(
            "DendroPy is required to read CASTER annotated support trees"
        ) from exc

    tree = dendropy.Tree.get(
        path=str(path),
        schema="newick",
        preserve_underscores=True,
        suppress_internal_node_taxa=True,
    )
    name_to_id: dict[str, int] = {}
    for leaf in tree.leaf_node_iter():
        label = leaf.taxon.label
        if label not in name_to_id:
            name_to_id[label] = len(name_to_id)

    result: dict[str, BranchSupport] = {}
    for node in tree.preorder_node_iter():
        if node is tree.seed_node or node.is_leaf():
            continue
        attrs = caster_label_attrs(node.label)
        if not {"s1", "s2", "s3"} <= attrs.keys():
            continue
        split_key = _canonical_split(
            (name_to_id[leaf.taxon.label] for leaf in node.leaf_iter()),
            len(name_to_id),
        )
        result[split_key] = _support_from_scores(
            (attrs["s1"], attrs["s2"], attrs["s3"])
        )
    if not result:
        raise ValueError(f"{path}: no internal branches with raw s1/s2/s3 annotations")
    return result


def _read_rows(path: Path, required: list[str]) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        fields = list(reader.fieldnames or [])
        missing = set(required) - set(fields)
        if missing:
            raise ValueError(f"{path}: missing columns: {', '.join(sorted(missing))}")
        return fields, list(reader)


def _format_float(value: float) -> str:
    return f"{value:.12g}"


def _mean(total: float, count: int) -> float:
    return total / count if count else math.nan


def _sequence_conflict(
    scores: tuple[float, float, float],
    q_locus: float,
    score_transform: str,
    epsilon: float,
) -> tuple[tuple[float, float, float], float, float, bool]:
    proportions = caster_score_proportions(scores)
    score_sum = sum(scores)
    evidence_weight = score_sum / q_locus if q_locus > 0.0 and score_sum > 0.0 else 0.0
    if q_locus <= 0.0:
        return proportions, evidence_weight, 0.0, False
    if score_transform == "mass":
        return proportions, evidence_weight, proportions[1] + proportions[2], True
    if score_transform == "raw-alt":
        conflict = (scores[1] + scores[2]) / q_locus if score_sum > 0.0 else 0.0
        return proportions, evidence_weight, conflict, True

    span = max(scores) - min(scores)
    if span <= epsilon:
        return proportions, evidence_weight, 0.0, False
    if score_transform == "winner":
        conflict = 1.0 if max(scores[1], scores[2]) > scores[0] else 0.0
    elif score_transform == "gap":
        conflict = (max(scores[1], scores[2]) - scores[0]) / span
        conflict = min(1.0, max(0.0, conflict))
    else:
        raise ValueError(f"unsupported score_transform: {score_transform}")
    return proportions, evidence_weight, conflict, True


def materialize_gene_scores(
    statistics_path: Path,
    support_tree: Path,
    score_output_path: Path,
    *,
    n_gene_trees: int,
) -> None:
    """Apply CASTER branch support to compact gene-tree statistics."""
    if n_gene_trees < 1:
        raise ValueError("n_gene_trees must be positive")
    support = caster_support_by_split(support_tree)
    aggregates = [GeneAggregate() for _ in range(n_gene_trees)]
    with statistics_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        missing = set(GENE_STATISTICS_FIELDS) - set(reader.fieldnames or [])
        if missing:
            raise ValueError(
                f"{statistics_path}: missing columns: {', '.join(sorted(missing))}"
            )
        for row in reader:
            gene_id = int(row["gene_id"])
            if not 0 <= gene_id < n_gene_trees:
                raise ValueError(f"{statistics_path}: gene_id out of range: {gene_id}")
            branch_support = support.get(row["split_key"])
            if branch_support is None:
                raise ValueError(
                    f"{support_tree}: support missing for split {row['split_key']}"
                )
            discordance = float(row["discordance_mass"])
            if not math.isfinite(discordance):
                continue
            aggregate = aggregates[gene_id]
            branch_loss = discordance * branch_support.penalty
            aggregate.n_valid_branches += 1
            aggregate.discordance_mass_sum += discordance
            aggregate.loss_norm_sum += branch_loss
            aggregate.loss_weighted_norm_sum += (
                branch_support.confidence * branch_loss
            )
            aggregate.sum_ils_ratio += branch_support.ils_ratio
            aggregate.sum_branch_confidence += branch_support.confidence

    with score_output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=GENE_SCORE_FIELDS,
            delimiter="\t",
            lineterminator="\n",
        )
        writer.writeheader()
        for gene_id, aggregate in enumerate(aggregates):
            count = aggregate.n_valid_branches
            writer.writerow(
                {
                    "gene_id": gene_id,
                    "n_valid_branches": count,
                    "discordance_mass_sum": _format_float(
                        aggregate.discordance_mass_sum
                    ),
                    "discordance_mass_mean": _format_float(
                        _mean(aggregate.discordance_mass_sum, count)
                    ),
                    "loss_norm_sum": _format_float(aggregate.loss_norm_sum),
                    "loss_norm_mean": _format_float(
                        _mean(aggregate.loss_norm_sum, count)
                    ),
                    "loss_weighted_norm_sum": _format_float(
                        aggregate.loss_weighted_norm_sum
                    ),
                    "loss_weighted_norm_mean": _format_float(
                        _mean(aggregate.loss_weighted_norm_sum, count)
                    ),
                    "mean_ils_ratio": _format_float(
                        _mean(aggregate.sum_ils_ratio, count)
                    ),
                    "mean_branch_confidence": _format_float(
                        _mean(aggregate.sum_branch_confidence, count)
                    ),
                }
            )


MSA_BRANCH_FIELDS = [
    "locus_id",
    "alignment_path",
    "species_branch_id",
    "split_key",
    "sequence_p1",
    "sequence_p2",
    "sequence_p3",
    "species_p1",
    "species_p2",
    "species_p3",
    "sequence_score1",
    "sequence_score2",
    "sequence_score3",
    "sequence_score_sum",
    "q_locus",
    "evidence_weight",
    "branch_loss",
    "branch_weighted_loss",
    "n_sites_used",
    "n_taxa_present",
    "support_source",
    "score_transform",
]

MSA_SCORE_FIELDS = [
    "locus_id",
    "alignment_path",
    "n_taxa_present",
    "n_sites_raw",
    "n_sites_used",
    "n_internal_gene_blocks",
    "n_valid_branches",
    "n_skipped_branches",
    "loss_norm_sum",
    "loss_norm_mean",
    "loss_weighted_norm_sum",
    "loss_weighted_norm_mean",
    "alt_winning_branch_fraction",
    "mean_ils_ratio",
    "mean_branch_confidence",
    "support_source",
    "score_transform",
]


def materialize_msa_scores(
    statistics_path: Path,
    item_path: Path,
    support_tree: Path,
    branch_output_path: Path,
    score_output_path: Path,
    *,
    score_transform: str,
    epsilon: float = 1e-12,
) -> None:
    if score_transform not in {"mass", "raw-alt", "gap", "winner"}:
        raise ValueError(f"unsupported score_transform: {score_transform}")

    _, item_rows = _read_rows(item_path, MSA_ITEM_FIELDS)
    aggregates: dict[str, LocusAggregate] = {}
    for row in item_rows:
        locus_id = row["locus_id"]
        n_taxa = int(row["n_taxa_present"])
        n_sites = int(row["n_sites_used"])
        n_blocks = int(row["n_internal_gene_blocks"])
        preliminary_skip = (
            n_taxa < 4
            or n_blocks == 0
            or (score_transform not in {"mass", "raw-alt"} and n_sites <= 0)
        )
        aggregates[locus_id] = LocusAggregate(
            metadata=row,
            preliminary_skip=preliminary_skip,
            n_skipped_branches=(int(row["n_species_branches"]) if preliminary_skip else 0),
        )

    support = caster_support_by_split(support_tree)
    with statistics_path.open(newline="", encoding="utf-8") as source, branch_output_path.open(
        "w", newline="", encoding="utf-8"
    ) as branch_handle:
        reader = csv.DictReader(source, delimiter="\t")
        missing = set(MSA_STATISTICS_FIELDS) - set(reader.fieldnames or [])
        if missing:
            raise ValueError(
                f"{statistics_path}: missing columns: {', '.join(sorted(missing))}"
            )
        writer = csv.DictWriter(
            branch_handle,
            fieldnames=MSA_BRANCH_FIELDS,
            delimiter="\t",
            lineterminator="\n",
        )
        writer.writeheader()
        for row in reader:
            aggregate = aggregates.get(row["locus_id"])
            if aggregate is None:
                raise ValueError(
                    f"{statistics_path}: unknown locus_id={row['locus_id']}"
                )
            metadata = aggregate.metadata
            if aggregate.preliminary_skip:
                continue
            branch_support = support.get(row["split_key"])
            if branch_support is None:
                raise ValueError(
                    f"{support_tree}: support missing for split {row['split_key']}"
                )
            scores = tuple(float(row[f"sequence_score{i}"]) for i in range(1, 4))
            q_locus = float(row["q_locus"])
            sequence_p, evidence_weight, conflict, has_score = _sequence_conflict(
                scores, q_locus, score_transform, epsilon
            )
            branch_loss = 0.0
            if has_score:
                if score_transform == "mass":
                    branch_loss = conflict * evidence_weight * branch_support.penalty
                else:
                    branch_loss = conflict * branch_support.penalty
            branch_weighted_loss = branch_support.confidence * branch_loss
            writer.writerow(
                {
                    **row,
                    "sequence_p1": _format_float(sequence_p[0]),
                    "sequence_p2": _format_float(sequence_p[1]),
                    "sequence_p3": _format_float(sequence_p[2]),
                    "species_p1": _format_float(branch_support.proportions[0]),
                    "species_p2": _format_float(branch_support.proportions[1]),
                    "species_p3": _format_float(branch_support.proportions[2]),
                    "evidence_weight": _format_float(evidence_weight),
                    "branch_loss": _format_float(branch_loss),
                    "branch_weighted_loss": _format_float(branch_weighted_loss),
                    "support_source": SUPPORT_SOURCE,
                    "score_transform": score_transform,
                }
            )
            if not has_score:
                aggregate.n_skipped_branches += 1
                continue
            aggregate.n_valid_branches += 1
            aggregate.loss_norm_sum += branch_loss
            aggregate.loss_weighted_norm_sum += branch_weighted_loss
            aggregate.alt_winning_branches += int(max(scores[1], scores[2]) > scores[0])
            aggregate.sum_ils_ratio += branch_support.ils_ratio
            aggregate.sum_branch_confidence += branch_support.confidence

    with score_output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=MSA_SCORE_FIELDS,
            delimiter="\t",
            lineterminator="\n",
        )
        writer.writeheader()
        for row in item_rows:
            aggregate = aggregates[row["locus_id"]]
            n_valid = aggregate.n_valid_branches
            writer.writerow(
                {
                    "locus_id": row["locus_id"],
                    "alignment_path": row["alignment_path"],
                    "n_taxa_present": row["n_taxa_present"],
                    "n_sites_raw": row["n_sites_raw"],
                    "n_sites_used": row["n_sites_used"],
                    "n_internal_gene_blocks": row["n_internal_gene_blocks"],
                    "n_valid_branches": n_valid,
                    "n_skipped_branches": aggregate.n_skipped_branches,
                    "loss_norm_sum": _format_float(aggregate.loss_norm_sum),
                    "loss_norm_mean": _format_float(
                        _mean(aggregate.loss_norm_sum, n_valid)
                    ),
                    "loss_weighted_norm_sum": _format_float(
                        aggregate.loss_weighted_norm_sum
                    ),
                    "loss_weighted_norm_mean": _format_float(
                        _mean(aggregate.loss_weighted_norm_sum, n_valid)
                    ),
                    "alt_winning_branch_fraction": _format_float(
                        _mean(float(aggregate.alt_winning_branches), n_valid)
                    ),
                    "mean_ils_ratio": _format_float(
                        _mean(aggregate.sum_ils_ratio, n_valid)
                    ),
                    "mean_branch_confidence": _format_float(
                        _mean(aggregate.sum_branch_confidence, n_valid)
                    ),
                    "support_source": SUPPORT_SOURCE,
                    "score_transform": score_transform,
                }
            )
