#!/usr/bin/env python3
"""Behavioral tests for independent weighted-Bernoulli tail calculations."""

from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

from scipy.stats import binom

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from iilslib.weighted_poisson_binomial import (  # noqa: E402
    WeightedBernoulliDistribution,
)


class WeightedPoissonBinomialTest(unittest.TestCase):
    def test_exact_moments_and_strict_tail_quantile(self) -> None:
        distribution = WeightedBernoulliDistribution([1.0, 2.0], [0.2, 0.1])

        self.assertEqual(distribution.method, "exact_enumeration")
        self.assertAlmostEqual(distribution.summary.mean, 0.4)
        self.assertAlmostEqual(distribution.summary.variance, 0.52)
        self.assertAlmostEqual(distribution.survival(1.0), 0.1)
        self.assertAlmostEqual(distribution.survival(1.5), 0.1)
        self.assertAlmostEqual(distribution.survival(2.0), 0.02)

        threshold = distribution.quantile(0.95)
        self.assertEqual(threshold.threshold, 2.0)
        self.assertAlmostEqual(threshold.tail_probability, 0.02)
        self.assertLessEqual(
            distribution.survival(threshold.threshold),
            0.05,
        )
        self.assertGreater(distribution.survival(1.0), 0.05)

    def test_deterministic_terms_produce_deterministic_threshold(self) -> None:
        distribution = WeightedBernoulliDistribution([2.0, 3.0], [1.0, 0.0])

        self.assertEqual(distribution.method, "degenerate")
        self.assertEqual(distribution.summary.mean, 2.0)
        self.assertEqual(distribution.summary.variance, 0.0)
        self.assertEqual(distribution.survival(1.999), 1.0)
        self.assertEqual(distribution.survival(2.0), 0.0)
        self.assertEqual(distribution.quantile(0.999).threshold, 2.0)

    def test_saddlepoint_tail_tracks_equal_weight_binomial(self) -> None:
        n_terms = 20
        probability = 0.2
        weight = 0.01
        distribution = WeightedBernoulliDistribution(
            [weight] * n_terms,
            [probability] * n_terms,
        )

        self.assertEqual(distribution.method, "lugannani_rice")
        threshold = distribution.quantile(0.99)
        exact_strict_tail = float(
            binom.sf(math.floor(threshold.threshold / weight + 1e-12), n_terms, probability)
        )
        self.assertLess(abs(exact_strict_tail - 0.01), 0.003)
        self.assertLessEqual(distribution.summary.mean, threshold.threshold)
        self.assertLessEqual(threshold.threshold, distribution.summary.support_max)

    def test_positive_rescaling_preserves_standardized_quantile(self) -> None:
        base = WeightedBernoulliDistribution([0.2] * 20, [0.15] * 20)
        scaled = WeightedBernoulliDistribution([0.3] * 20, [0.15] * 20)

        base_threshold = base.quantile(0.999).threshold
        scaled_threshold = scaled.quantile(0.999).threshold
        base_z = (base_threshold - base.summary.mean) / base.summary.standard_deviation
        scaled_z = (
            scaled_threshold - scaled.summary.mean
        ) / scaled.summary.standard_deviation
        self.assertAlmostEqual(base_z, scaled_z, places=9)

    def test_rejects_invalid_term_contracts(self) -> None:
        with self.assertRaisesRegex(ValueError, "same length"):
            WeightedBernoulliDistribution([1.0], [0.1, 0.2])
        with self.assertRaisesRegex(ValueError, "nonnegative"):
            WeightedBernoulliDistribution([-1.0], [0.1])
        with self.assertRaisesRegex(ValueError, "between zero and one"):
            WeightedBernoulliDistribution([1.0], [1.1])


if __name__ == "__main__":
    unittest.main()
