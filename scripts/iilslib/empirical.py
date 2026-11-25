"""Exact decimal probabilities for finite, equally weighted sample quantiles.

Distribution quantiles with probability masses must not use this module.
"""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from fractions import Fraction

import numpy as np

ALGORITHM = "empirical-order-statistic-decimal-ceil-v1"


def decimal_probability(value: str | Decimal) -> Decimal:
    if not isinstance(value, (str, Decimal)):
        raise TypeError("probability must retain its original decimal text")
    try:
        q = Decimal(value)
    except InvalidOperation as exc:
        raise ValueError("probability must be a finite decimal in [0, 1]") from exc
    if not q.is_finite() or not Decimal(0) <= q <= Decimal(1):
        raise ValueError("probability must be a finite decimal in [0, 1]")
    return q


def probability_text(value: str | Decimal) -> str:
    q = decimal_probability(value)
    if not q:
        return '0'
    # Decimal.normalize() obeys context precision and can round a long input.
    text = format(q, 'f')
    return text.rstrip('0').rstrip('.') if '.' in text else text


def empirical_rank(q: str | Decimal, n: int) -> int:
    if not isinstance(n, (int, np.integer)) or isinstance(n, bool) or n <= 0:
        raise ValueError("sample size must be a positive integer")
    a, b = Fraction(decimal_probability(q)).as_integer_ratio()
    return max(1, (a * int(n) + b - 1) // b)


@dataclass(frozen=True, slots=True)
class EmpiricalResult:
    threshold: float
    q: str
    n: int
    rank: int
    ties: int
    strict_exceedance_count: int
    algorithm: str = ALGORITHM


def empirical_quantile(values, q: str | Decimal, *, inplace: bool = False) -> EmpiricalResult:
    samples = np.asarray(values, dtype=np.float64)
    if samples.ndim != 1 or samples.size == 0 or not np.all(np.isfinite(samples)):
        raise ValueError("empirical samples must be a nonempty finite vector")
    rank = empirical_rank(q, samples.size)
    if not inplace:
        samples = samples.copy()
    if not samples.flags.writeable:
        raise ValueError("inplace selection requires writable samples")
    samples.partition(rank - 1)
    threshold = float(samples[rank - 1])
    return EmpiricalResult(threshold, probability_text(q), int(samples.size), rank,
                           int(np.count_nonzero(samples == threshold)),
                           int(np.count_nonzero(samples > threshold)))
