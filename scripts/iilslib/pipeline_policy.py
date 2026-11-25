from __future__ import annotations

from dataclasses import dataclass, replace
from decimal import Decimal
from enum import StrEnum, unique
from typing import assert_never

from .empirical import decimal_probability, probability_text


@unique
class ThresholdPolicy(StrEnum):
    LOCUS = "locus"
    STANDARDIZED_QUANTILE = "standardized-quantile"
    STANDARDIZED_POOLED = "standardized-pooled"
    POOLED_TAIL = "pooled-tail"

    @property
    def serialized(self) -> str:
        match self:
            case ThresholdPolicy.LOCUS:
                return "locus"
            case ThresholdPolicy.STANDARDIZED_QUANTILE:
                return "standardized_quantile"
            case ThresholdPolicy.STANDARDIZED_POOLED:
                return "standardized_pooled"
            case ThresholdPolicy.POOLED_TAIL:
                return "pooled_tail"
            case unreachable:
                assert_never(unreachable)


@dataclass(frozen=True, slots=True)
class ThresholdPolicyInput:
    threshold_policy: ThresholdPolicy | str
    threshold_policy_quantile: str
    threshold_policy_sample_size: int
    threshold_policy_sample_seed: int
    threshold_quantile: str
    threshold_label: str
    threshold_source: str
    score_transform: str
    filter_mode: str
    threshold_column: str


@dataclass(frozen=True, slots=True)
class ParsedThresholdPolicy:
    policy: ThresholdPolicy
    policy_quantile: Decimal
    sample_size: int
    sample_seed: int
    threshold_quantile: Decimal
    threshold_label: str
    threshold_source: str
    score_transform: str
    filter_mode: str
    threshold_column: str
    canonical_pooled_label: str | None = None


def parse_threshold_policy(args: ThresholdPolicyInput) -> ParsedThresholdPolicy:
    return ParsedThresholdPolicy(
        policy=ThresholdPolicy(args.threshold_policy),
        policy_quantile=decimal_probability(args.threshold_policy_quantile),
        sample_size=args.threshold_policy_sample_size,
        sample_seed=args.threshold_policy_sample_seed,
        threshold_quantile=decimal_probability(args.threshold_quantile),
        threshold_label=args.threshold_label,
        threshold_source=args.threshold_source,
        score_transform=args.score_transform,
        filter_mode=args.filter_mode,
        threshold_column=args.threshold_column,
    )


def threshold_quantile_label(value: str | Decimal) -> str:
    text = probability_text(value)
    return "q" + (text if "." not in text else text.split(".", 1)[1])


def validate_threshold_policy(
    parsed: ParsedThresholdPolicy,
) -> ParsedThresholdPolicy:
    if not 0.0 <= parsed.policy_quantile <= 1.0:
        raise ValueError("--threshold-policy-quantile must be between zero and one")
    match parsed.policy:
        case ThresholdPolicy.STANDARDIZED_QUANTILE:
            pass
        case ThresholdPolicy.STANDARDIZED_POOLED:
            if parsed.policy_quantile != 0.0:
                raise ValueError(
                    "--threshold-policy-quantile must be zero for standardized-pooled"
                )
        case ThresholdPolicy.LOCUS | ThresholdPolicy.POOLED_TAIL:
            if parsed.policy_quantile != 0.0:
                raise ValueError(
                    "--threshold-policy-quantile requires "
                    "--threshold-policy standardized-quantile"
                )
        case unreachable:
            assert_never(unreachable)
    if parsed.sample_size < 0:
        raise ValueError("--threshold-policy-sample-size must be nonnegative")
    if parsed.sample_seed < 0:
        raise ValueError("--threshold-policy-sample-seed must be nonnegative")
    if parsed.policy is ThresholdPolicy.LOCUS and parsed.sample_size != 0:
        raise ValueError(
            "--threshold-policy-sample-size applies only to "
            "standardized-quantile or pooled-tail"
        )
    match parsed.policy:
        case ThresholdPolicy.STANDARDIZED_POOLED:
            if parsed.threshold_source != "gene-tree-sim":
                raise ValueError(
                    "standardized-pooled requires --threshold-source gene-tree-sim"
                )
            if parsed.score_transform != "raw-alt":
                raise ValueError(
                    "standardized-pooled requires --score-transform raw-alt"
                )
            canonical_label = threshold_quantile_label(parsed.threshold_quantile)
            if (
                parsed.filter_mode == "threshold"
                and parsed.threshold_column != "threshold_score"
            ):
                raise ValueError(
                    "standardized-pooled filtering requires "
                    "--threshold-column threshold_score"
                )
            return replace(parsed, canonical_pooled_label=parsed.threshold_label or canonical_label)
        case ThresholdPolicy.POOLED_TAIL:
            if parsed.threshold_source == "gene-tree-sim":
                raise ValueError(
                    "--threshold-policy pooled-tail is currently available only with "
                    "--threshold-source analytic-pb"
                )
        case ThresholdPolicy.LOCUS | ThresholdPolicy.STANDARDIZED_QUANTILE:
            pass
        case unreachable:
            assert_never(unreachable)
    return parsed


def supports_threshold(parsed: ParsedThresholdPolicy) -> bool:
    if parsed.threshold_source == "gene-tree-sim":
        match parsed.policy:
            case (
                ThresholdPolicy.STANDARDIZED_QUANTILE
                | ThresholdPolicy.STANDARDIZED_POOLED
            ):
                return parsed.score_transform == "raw-alt"
            case ThresholdPolicy.LOCUS | ThresholdPolicy.POOLED_TAIL:
                return parsed.score_transform in {"mass", "raw-alt"}
            case unreachable:
                assert_never(unreachable)
    return parsed.score_transform == "raw-alt"


def analytic_policy_label(parsed: ParsedThresholdPolicy) -> str:
    match parsed.policy:
        case ThresholdPolicy.LOCUS:
            return "locus"
        case ThresholdPolicy.POOLED_TAIL:
            label = "pooled"
        case ThresholdPolicy.STANDARDIZED_POOLED:
            return f"pooled_n{parsed.sample_size}_s{parsed.sample_seed}"
        case ThresholdPolicy.STANDARDIZED_QUANTILE:
            if parsed.policy_quantile == 0.0:
                label = "zmin"
            elif parsed.policy_quantile == 1.0:
                label = "zmax"
            else:
                value = probability_text(parsed.policy_quantile)
                label = "zq" + value.split(".", 1)[1]
        case unreachable:
            assert_never(unreachable)
    if parsed.sample_size > 0:
        label += f"_n{parsed.sample_size}_s{parsed.sample_seed}"
    return label


def filter_label(parsed: ParsedThresholdPolicy) -> str:
    if parsed.filter_mode == "threshold":
        policy_label = analytic_policy_label(parsed)
        if parsed.threshold_source == "analytic-pb":
            return f"analytic_pb_{policy_label}_{parsed.threshold_label}"
        match parsed.policy:
            case ThresholdPolicy.LOCUS:
                return f"gene_tree_{parsed.threshold_label}"
            case ThresholdPolicy.STANDARDIZED_POOLED:
                pooled_label = (
                    parsed.canonical_pooled_label or parsed.threshold_label
                )
                return f"gene_tree_{policy_label}_{pooled_label}"
            case ThresholdPolicy.STANDARDIZED_QUANTILE | ThresholdPolicy.POOLED_TAIL:
                return f"gene_tree_{policy_label}_{parsed.threshold_label}"
            case unreachable:
                assert_never(unreachable)
    return "quantile_q95"
