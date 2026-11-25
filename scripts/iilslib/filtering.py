from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path
from statistics import median

from .tsv import read_tsv


@dataclass
class ScoreRow:
    order: int
    locus_id: str
    alignment_path: str
    score_text: str
    score: float
    is_finite: bool
    score_transform: str = ""
    score_transform_present: bool = False
    decision: str = "keep"
    reason: str = "finite_not_top_loss"
    rank_descending_loss: int | None = None
    finite_percentile: float | None = None
    threshold_score_text: str = ""
    threshold_label: str = ""
    threshold_quantile: str = ""
    threshold_score_transform: str = ""
    threshold_source: str = ""
    threshold_policy: str = ""
    threshold_tail_method: str = ""
    threshold_policy_sampled: str = ""
    threshold_policy_sample_rank: str = ""
    threshold_policy_sample_size_requested: str = ""
    threshold_policy_sample_size_used: str = ""
    threshold_policy_sample_seed: str = ""


def parse_float(text: str) -> float:
    try:
        return float(text)
    except (TypeError, ValueError):
        return math.nan


def format_float(value: float) -> str:
    return f"{value:.12g}" if math.isfinite(value) else ""


def quantile(values: list[float], p: float) -> float:
    if not values:
        return math.nan
    if len(values) == 1:
        return values[0]
    ordered = sorted(values)
    pos = (len(ordered) - 1) * p
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return ordered[lo]
    weight = pos - lo
    return ordered[lo] * (1 - weight) + ordered[hi] * weight


def median_absolute_deviation(values: list[float]) -> float:
    if not values:
        return math.nan
    med = median(values)
    return median(abs(value - med) for value in values)


def read_score_rows(path: Path, score_column: str) -> list[ScoreRow]:
    required = {"locus_id", "alignment_path", score_column}
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        fieldnames = reader.fieldnames or []
        missing = required - set(fieldnames)
        if missing:
            raise ValueError(
                f"{path} is missing required columns: {', '.join(sorted(missing))}"
            )
        has_score_transform = "score_transform" in fieldnames
        parsed: list[ScoreRow] = []
        for order, row in enumerate(reader):
            score_text = row.get(score_column, "nan")
            score = parse_float(score_text)
            parsed.append(
                ScoreRow(
                    order=order,
                    locus_id=row["locus_id"],
                    alignment_path=row["alignment_path"],
                    score_text=score_text,
                    score=score,
                    is_finite=math.isfinite(score),
                    score_transform=row.get("score_transform", ""),
                    score_transform_present=has_score_transform,
                )
            )
    return parsed


def apply_high_loss_filter(rows: list[ScoreRow], drop_fraction: float) -> dict[str, str]:
    """Remove non-finite scores and the high-loss finite tail in place."""
    if not 0 <= drop_fraction < 1:
        raise ValueError("--drop-fraction must be in [0, 1)")

    for row in rows:
        if not row.is_finite:
            row.decision = "remove"
            row.reason = "nonfinite_score"

    finite_rows = [row for row in rows if row.is_finite]
    finite_rows.sort(key=lambda row: (-row.score, row.order))
    n_drop = math.ceil(len(finite_rows) * drop_fraction)

    for rank, row in enumerate(finite_rows, start=1):
        row.rank_descending_loss = rank
        row.finite_percentile = 1.0 - (rank - 1) / len(finite_rows)
        if rank <= n_drop:
            row.decision = "remove"
            row.reason = "top_loss_fraction"

    finite_scores = [row.score for row in rows if row.is_finite]
    removed = [row for row in rows if row.decision == "remove"]
    kept = [row for row in rows if row.decision == "keep"]
    nonfinite = [row for row in rows if not row.is_finite]
    threshold = finite_rows[n_drop - 1].score if n_drop > 0 and finite_rows else math.nan

    return {
        "n_total": str(len(rows)),
        "n_finite": str(len(finite_scores)),
        "n_nonfinite_removed": str(len(nonfinite)),
        "drop_fraction": f"{drop_fraction:.12g}",
        "n_top_loss_removed": str(n_drop),
        "n_removed_total": str(len(removed)),
        "n_kept": str(len(kept)),
        "threshold_score": format_float(threshold),
        "finite_min": format_float(min(finite_scores)) if finite_scores else "",
        "finite_q25": format_float(quantile(finite_scores, 0.25)),
        "finite_median": format_float(median(finite_scores)) if finite_scores else "",
        "finite_q75": format_float(quantile(finite_scores, 0.75)),
        "finite_q95": format_float(quantile(finite_scores, 0.95)),
        "finite_max": format_float(max(finite_scores)) if finite_scores else "",
        "finite_mad": format_float(median_absolute_deviation(finite_scores)),
    }
