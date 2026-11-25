#!/usr/bin/env python3
"""Contracts for support interpretation outside the C++ statistics interface."""

from __future__ import annotations

import csv
import math
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from iilslib.statistics import (  # noqa: E402
    caster_score_proportions,
)


class CasterAnnotatedSupportTransformTest(unittest.TestCase):
    def test_raw_caster_scores_are_reconstructed_as_msc_proportions(self) -> None:
        actual = caster_score_proportions((9.0, 3.0, 1.0))
        for observed, expected in zip(actual, (0.6, 0.2, 0.2)):
            self.assertAlmostEqual(observed, expected, places=12)

    def test_nonpositive_runner_up_is_high_confidence(self) -> None:
        self.assertEqual(caster_score_proportions((9.0, 0.0, -1.0)), (1.0, 0.0, 0.0))

    def test_gene_tree_materialization_cli_uses_compact_native_schema(self) -> None:
        with tempfile.TemporaryDirectory(prefix="iils_gene_materialize_") as tmp:
            tmpdir = Path(tmp)
            statistics = tmpdir / "statistics.tsv"
            support = tmpdir / "support.tre"
            scores = tmpdir / "scores.tsv"
            statistics.write_text(
                "gene_id\tsplit_key\tdiscordance_mass\n"
                "0\t0,1\t0\n"
                "1\t0,1\t1\n"
                "2\t0,1\t0\n",
                encoding="utf-8",
            )
            support.write_text(
                "((A,B)'[s1=9;s2=3;s3=1]',(C,D));\n",
                encoding="utf-8",
            )

            subprocess.run(
                [
                    sys.executable,
                    str(SCRIPTS / "materialize_iils_statistics.py"),
                    "--input-type",
                    "gene-tree",
                    "--statistics",
                    str(statistics),
                    "--support-tree",
                    str(support),
                    "--n-gene-trees",
                    "3",
                    "--score-output",
                    str(scores),
                ],
                check=True,
                capture_output=True,
                text=True,
            )

            with scores.open(newline="", encoding="utf-8") as handle:
                reader = csv.DictReader(handle, delimiter="\t")
                rows = list(reader)
            self.assertEqual(
                reader.fieldnames,
                [
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
                ],
            )
            self.assertEqual(float(rows[0]["loss_norm_mean"]), 0.0)
            self.assertAlmostEqual(
                float(rows[1]["loss_norm_mean"]),
                -math.log(0.6),
                places=11,
            )
            self.assertEqual(float(rows[1]["discordance_mass_mean"]), 1.0)

if __name__ == "__main__":
    unittest.main()
