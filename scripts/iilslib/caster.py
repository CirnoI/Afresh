from __future__ import annotations

import math
import re
from pathlib import Path


CASTER_RAW_S_FORMULA = (
    "max(0.0, -log(1.5 - 1.5 * s_best / (s_best + 2*s_second)))"
)


def validate_name_map(path: Path) -> None:
    """Validate a CASTER/Afresh two-column sequence-to-species map."""
    if not path.is_file() or path.stat().st_size == 0:
        raise ValueError(f"name map is missing or empty: {path}")
    seen: dict[str, str] = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, 1):
            fields = raw.split()
            if len(fields) != 2:
                raise ValueError(
                    f"{path}:{line_number}: name map rows must contain exactly two fields"
                )
            sequence_name, species_name = fields
            if sequence_name in seen:
                previous = seen[sequence_name]
                if previous == species_name:
                    raise ValueError(
                        f"{path}:{line_number}: duplicate sequence name {sequence_name!r}"
                    )
                raise ValueError(
                    f"{path}:{line_number}: conflicting mapping for {sequence_name!r} "
                    f"({previous!r} vs {species_name!r})"
                )
            seen[sequence_name] = species_name


def has_caster_support_annotation(path: Path) -> bool:
    """Return true when a CASTER -u 2 tree contains raw s1/s2/s3 scores."""
    if not path.is_file() or path.stat().st_size == 0:
        return False
    text = path.read_text(encoding="utf-8", errors="ignore")
    return "s1=" in text and "s2=" in text and "s3=" in text


def caster_label_attrs(label: str | None) -> dict[str, float]:
    """Parse numeric key=value attributes from a CASTER Newick node label."""
    if not label:
        return {}
    text = label.strip().strip("'\"")
    if text.startswith("[") and text.endswith("]"):
        text = text[1:-1]
    attrs: dict[str, float] = {}
    for key, value in re.findall(r"(?:^|;)([A-Za-z_][A-Za-z0-9_]*)=([^;\]]+)", text):
        try:
            attrs[key] = float(value)
        except ValueError:
            continue
    return attrs


def caster_raw_s_scores(label: str | None) -> tuple[float, float, float] | None:
    """Return raw CASTER s1/s2/s3 scores from a Newick node label."""
    attrs = caster_label_attrs(label)
    if not {"s1", "s2", "s3"} <= set(attrs):
        return None
    return attrs["s1"], attrs["s2"], attrs["s3"]


def caster_raw_s_cu_length(
    scores: tuple[float, float, float],
    *,
    infinite_length: float | None = None,
) -> tuple[float, bool]:
    """Convert raw CASTER topology scores to an MSC-shaped CU branch length.

    The three raw scores are sorted before applying the formula, so the
    calculation uses the highest and second-highest CASTER topology scores.
    Infinite-limit cases can be returned as math.inf or replaced by
    infinite_length for downstream simulators that require finite edge lengths.
    """
    if any(not math.isfinite(score) for score in scores):
        raise ValueError(f"non-finite CASTER raw scores: {scores}")
    best, second, _third = sorted(scores, reverse=True)
    if best <= 0.0:
        return 0.0, False
    if second <= 0.0:
        return (infinite_length if infinite_length is not None else math.inf), True

    exp_minus_t = 1.5 - 1.5 * best / (best + 2.0 * second)
    if exp_minus_t <= 0.0:
        return (infinite_length if infinite_length is not None else math.inf), True
    return max(0.0, -math.log(exp_minus_t)), False


def stderr_tail(path: Path, n_lines: int = 20) -> str:
    """Return a short tail from a stderr file for diagnostic exceptions."""
    if not path.is_file() or path.stat().st_size == 0:
        return ""
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    return "\n".join(lines[-n_lines:])


def caster_site_command(
    caster_bin: Path,
    loci_file: Path,
    output_tree: Path,
    threads: int,
    *,
    detail_level: int | None = 2,
    length_unit: str | None = "CULength",
    constraint_tree: Path | None = None,
    mapping_path: Path | None = None,
    score_constraint_only: bool = False,
    seed: int | None = None,
) -> list[str | Path]:
    """Build a CASTER-site command for a one-entry-per-line MSA list."""
    if threads < 1:
        raise ValueError("threads must be a positive integer")
    if score_constraint_only and constraint_tree is None:
        raise ValueError("score_constraint_only requires constraint_tree")
    if seed is not None and seed < 0:
        raise ValueError("seed must be nonnegative")

    cmd: list[str | Path] = [caster_bin]
    if score_constraint_only:
        cmd.append("-C")
    if constraint_tree is not None:
        cmd.extend(["-c", constraint_tree])
    if mapping_path is not None:
        cmd.extend(["-a", mapping_path])
    cmd.extend(["-f", "list"])
    if detail_level is not None:
        cmd.extend(["-u", str(detail_level)])
    if length_unit is not None:
        cmd.extend(["--length", length_unit])
    if seed is not None:
        cmd.extend(["--seed", str(seed)])
    cmd.extend(["-t", str(threads), "-o", output_tree, loci_file])
    return cmd
