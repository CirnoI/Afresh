from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy.optimize import brentq
from scipy.special import expit, ndtr


EXACT_ENUM_MAX_TERMS = 16
_ROOT_XTOL = 1e-13
_ROOT_RTOL = 1e-12


@dataclass(frozen=True)
class WeightedBernoulliSummary:
    mean: float
    variance: float
    standard_deviation: float
    skewness: float
    support_min: float
    support_max: float
    n_active_terms: int
    effective_term_count: float
    max_weight_share: float


@dataclass(frozen=True)
class TailResult:
    threshold: float
    tail_probability: float
    method: str
    iterations: int
    status: str


class WeightedBernoulliDistribution:
    """Distribution of a nonnegative weighted sum of independent Bernoulli terms."""

    def __init__(
        self,
        weights: Sequence[float] | np.ndarray,
        probabilities: Sequence[float] | np.ndarray,
        *,
        exact_max_terms: int = EXACT_ENUM_MAX_TERMS,
    ) -> None:
        raw_weights = np.asarray(weights, dtype=np.float64)
        raw_probabilities = np.asarray(probabilities, dtype=np.float64)
        if raw_weights.ndim != 1 or raw_probabilities.ndim != 1:
            raise ValueError("weights and probabilities must be one-dimensional")
        if raw_weights.shape != raw_probabilities.shape:
            raise ValueError("weights and probabilities must have the same length")
        if not np.all(np.isfinite(raw_weights)):
            raise ValueError("weights must be finite")
        if not np.all(np.isfinite(raw_probabilities)):
            raise ValueError("probabilities must be finite")
        if np.any(raw_weights < 0.0):
            raise ValueError("weights must be nonnegative")
        if np.any((raw_probabilities < 0.0) | (raw_probabilities > 1.0)):
            raise ValueError("probabilities must be between zero and one")
        if exact_max_terms < 0:
            raise ValueError("exact_max_terms must be nonnegative")

        positive = raw_weights > 0.0
        deterministic = positive & (raw_probabilities == 1.0)
        active = positive & (raw_probabilities > 0.0) & (raw_probabilities < 1.0)
        self._offset = float(np.sum(raw_weights[deterministic]))
        self._weights = np.ascontiguousarray(raw_weights[active])
        self._probabilities = np.ascontiguousarray(raw_probabilities[active])
        self._log_probabilities = np.log(self._probabilities)
        self._log_complements = np.log1p(-self._probabilities)
        self._exact_max_terms = exact_max_terms
        self._exact_scores: np.ndarray | None = None
        self._exact_tail_after: np.ndarray | None = None

        variance_terms = (
            self._weights**2 * self._probabilities * (1.0 - self._probabilities)
        )
        variance = float(np.sum(variance_terms))
        standard_deviation = math.sqrt(max(0.0, variance))
        third_cumulant = float(
            np.sum(
                self._weights**3
                * self._probabilities
                * (1.0 - self._probabilities)
                * (1.0 - 2.0 * self._probabilities)
            )
        )
        skewness = (
            third_cumulant / standard_deviation**3
            if standard_deviation > 0.0
            else 0.0
        )
        weight_sum = float(np.sum(self._weights))
        squared_weight_sum = float(np.sum(self._weights**2))
        effective_term_count = (
            weight_sum**2 / squared_weight_sum if squared_weight_sum > 0.0 else 0.0
        )
        max_weight_share = (
            float(np.max(self._weights)) / weight_sum if weight_sum > 0.0 else 0.0
        )
        self.summary = WeightedBernoulliSummary(
            mean=self._offset + float(np.dot(self._weights, self._probabilities)),
            variance=variance,
            standard_deviation=standard_deviation,
            skewness=skewness,
            support_min=self._offset,
            support_max=self._offset + weight_sum,
            n_active_terms=int(self._weights.size),
            effective_term_count=effective_term_count,
            max_weight_share=max_weight_share,
        )

    @property
    def method(self) -> str:
        if self.summary.variance == 0.0:
            return "degenerate"
        if self.summary.n_active_terms <= self._exact_max_terms:
            return "exact_enumeration"
        return "lugannani_rice"

    def _prepare_exact_distribution(self) -> None:
        if self._exact_scores is not None:
            return
        scores = np.array([self._offset], dtype=np.float64)
        masses = np.array([1.0], dtype=np.float64)
        for weight, probability in zip(
            self._weights, self._probabilities, strict=True
        ):
            previous_scores = scores
            previous_masses = masses
            scores = np.concatenate((previous_scores, previous_scores + weight))
            masses = np.concatenate(
                (
                    previous_masses * (1.0 - probability),
                    previous_masses * probability,
                )
            )
        order = np.argsort(scores, kind="mergesort")
        scores = scores[order]
        masses = masses[order]
        unique_scores, inverse = np.unique(scores, return_inverse=True)
        unique_masses = np.zeros(unique_scores.size, dtype=np.float64)
        np.add.at(unique_masses, inverse, masses)
        total = float(np.sum(unique_masses))
        if not math.isfinite(total) or total <= 0.0:
            raise ArithmeticError("invalid exact weighted-Bernoulli probability mass")
        unique_masses /= total
        self._exact_scores = unique_scores
        self._exact_tail_after = 1.0 - np.cumsum(unique_masses)
        self._exact_tail_after[-1] = 0.0

    def _tilted_statistics(self, tilt: float) -> tuple[float, float, float]:
        logits = self._log_probabilities - self._log_complements + tilt * self._weights
        tilted_probabilities = expit(logits)
        cumulant = float(
            np.sum(
                np.logaddexp(
                    self._log_complements,
                    self._log_probabilities + tilt * self._weights,
                )
            )
        )
        first = float(np.dot(self._weights, tilted_probabilities))
        second = float(
            np.sum(
                self._weights**2
                * tilted_probabilities
                * (1.0 - tilted_probabilities)
            )
        )
        return cumulant, first, second

    def _solve_tilt(self, centered_score: float) -> tuple[float, int]:
        centered_mean = self.summary.mean - self._offset
        if centered_score == centered_mean:
            return 0.0, 0
        evaluations = 0

        def objective(tilt: float) -> float:
            nonlocal evaluations
            evaluations += 1
            return self._tilted_statistics(tilt)[1] - centered_score

        if centered_score > centered_mean:
            lower = 0.0
            upper = 1.0
            while objective(upper) < 0.0:
                upper *= 2.0
                if upper > 1e16:
                    raise ArithmeticError("could not bracket positive saddlepoint tilt")
        else:
            lower = -1.0
            upper = 0.0
            while objective(lower) > 0.0:
                lower *= 2.0
                if lower < -1e16:
                    raise ArithmeticError("could not bracket negative saddlepoint tilt")
        root = brentq(
            objective,
            lower,
            upper,
            xtol=_ROOT_XTOL,
            rtol=_ROOT_RTOL,
            maxiter=200,
        )
        return float(root), evaluations

    def _lugannani_rice_survival(self, threshold: float) -> tuple[float, int]:
        centered_score = threshold - self._offset
        centered_mean = self.summary.mean - self._offset
        standard_deviation = self.summary.standard_deviation
        if standard_deviation == 0.0:
            return (1.0 if threshold < self.summary.mean else 0.0), 0
        if abs(centered_score - centered_mean) <= 1e-12 * max(
            1.0, standard_deviation
        ):
            return 0.5, 0

        tilt, evaluations = self._solve_tilt(centered_score)
        cumulant, _, second = self._tilted_statistics(tilt)
        deviance = max(0.0, 2.0 * (tilt * centered_score - cumulant))
        w = math.copysign(math.sqrt(deviance), tilt)
        u = tilt * math.sqrt(max(0.0, second))
        if abs(w) < 1e-10 or abs(u) < 1e-14:
            z = (threshold - self.summary.mean) / standard_deviation
            return float(ndtr(-z)), evaluations
        density = math.exp(-0.5 * w * w) / math.sqrt(2.0 * math.pi)
        survival = float(ndtr(-w)) + density * (1.0 / u - 1.0 / w)
        return min(1.0, max(0.0, survival)), evaluations

    def survival(self, threshold: float) -> float:
        """Return P(S > threshold), matching the pipeline's strict filter rule."""
        if not math.isfinite(threshold):
            if threshold == math.inf:
                return 0.0
            if threshold == -math.inf:
                return 1.0
            return math.nan
        if threshold < self.summary.support_min:
            return 1.0
        if threshold >= self.summary.support_max:
            return 0.0
        if self.method == "degenerate":
            return 1.0 if threshold < self.summary.mean else 0.0
        if self.method == "exact_enumeration":
            self._prepare_exact_distribution()
            assert self._exact_scores is not None
            assert self._exact_tail_after is not None
            index = int(np.searchsorted(self._exact_scores, threshold, side="right"))
            if index >= self._exact_scores.size:
                return 0.0
            return float(1.0 if index == 0 else self._exact_tail_after[index - 1])
        return self._lugannani_rice_survival(threshold)[0]

    def quantile(self, quantile: float) -> TailResult:
        """Return a threshold whose strict upper tail targets 1 - quantile."""
        if not 0.0 < quantile < 1.0:
            raise ValueError("quantile must be between zero and one")
        target_tail = 1.0 - quantile
        if self.method == "degenerate":
            return TailResult(
                threshold=self.summary.mean,
                tail_probability=0.0,
                method=self.method,
                iterations=0,
                status="completed",
            )
        if self.method == "exact_enumeration":
            self._prepare_exact_distribution()
            assert self._exact_scores is not None
            assert self._exact_tail_after is not None
            candidates = np.flatnonzero(self._exact_tail_after <= target_tail)
            index = int(candidates[0]) if candidates.size else self._exact_scores.size - 1
            return TailResult(
                threshold=float(self._exact_scores[index]),
                tail_probability=float(self._exact_tail_after[index]),
                method=self.method,
                iterations=0,
                status="completed",
            )

        lower = self.summary.support_min
        upper = self.summary.support_max
        evaluations = 0

        def objective(threshold: float) -> float:
            nonlocal evaluations
            if threshold <= self.summary.support_min:
                return 1.0 - target_tail
            if threshold >= self.summary.support_max:
                return -target_tail
            tail, tail_evaluations = self._lugannani_rice_survival(threshold)
            evaluations += 1 + tail_evaluations
            return tail - target_tail

        root = brentq(
            objective,
            lower,
            upper,
            xtol=_ROOT_XTOL,
            rtol=_ROOT_RTOL,
            maxiter=200,
        )
        tail = self.survival(float(root))
        return TailResult(
            threshold=float(root),
            tail_probability=tail,
            method=self.method,
            iterations=evaluations,
            status="completed",
        )
