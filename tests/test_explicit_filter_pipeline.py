"""Contracts for the explicit one-shot filtering entry point."""

from __future__ import annotations

import json
from contextlib import redirect_stderr
import io
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from iilslib.filtering import ScoreRow, apply_high_loss_filter
from iilslib.one_shot_config import inputs, normalize, parse_args
from iilslib.tsv import write_tsv


class ExplicitOneShotTest(unittest.TestCase):
    def base_args(self, root: Path, *extra: str):
        return parse_args([
            "--loci-file", str(root / "loci.txt"),
            "--guide-tree", str(root / "guide.nwk"),
            "--out-root", str(root / "out"),
            "--scratch-root", str(root / "scratch"),
            *extra,
        ])

    def make_inputs(self, root: Path) -> None:
        locus = root / "one.fa"
        locus.write_text(">A\nACGT\n>B\nACGT\n>C\nACGT\n>D\nACGT\n")
        (root / "loci.txt").write_text(str(locus) + "\n")
        (root / "guide.nwk").write_text("((A,B)'[s1=70;s2=20;s3=10]':0.5,(C,D));\n")

    def test_retired_cache_and_stage_arguments_fail_before_execution(self):
        with self.assertRaisesRegex(ValueError, "retired one-shot options"):
            parse_args(["--loci-file", "/tmp/loci", "--out-root", "/tmp/out",
                        "--scratch-root", "/tmp/scratch", "--cache-root", "/tmp/cache"])

    def test_tree_comparison_is_not_part_of_the_core_cli(self):
        with redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                parse_args(["--loci-file", "/tmp/loci", "--out-root", "/tmp/out",
                            "--scratch-root", "/tmp/scratch", "--compare"])

    def test_new_input_contract_and_seed_zero(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.make_inputs(root)
            args = self.base_args(root, "--seed", "0", "--threshold-quantile", "0.99")
            specs = normalize(args)
            runs = inputs(args)
            self.assertEqual(args.seed, 0)
            self.assertEqual(runs[0].run_id, "run")
            self.assertEqual(specs[0].q, "0.99")

    def test_old_batch_schema_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.make_inputs(root)
            manifest = root / "runs.tsv"
            write_tsv(manifest, ["run_id", "loci_file", "out_dir", "caster_tree"], [{
                "run_id": "run", "loci_file": str(root / "loci.txt"),
                "out_dir": str(root / "out"), "caster_tree": str(root / "tree.nwk"),
            }])
            args = parse_args(["--run-manifest", str(manifest), "--out-root", str(root / "out"),
                               "--scratch-root", str(root / "scratch")])
            normalize(args)
            with self.assertRaisesRegex(ValueError, "old run manifest schema"):
                inputs(args)

    def test_fixed_fraction_uses_ceil_and_stable_ties(self):
        rows = [ScoreRow(i, str(i), f"{i}.fa", "1", 1.0, True) for i in range(5)]
        summary = apply_high_loss_filter(rows, 0.2)
        self.assertEqual(summary["n_top_loss_removed"], "1")
        self.assertEqual([row.locus_id for row in rows if row.decision == "remove"], ["0"])

    def test_output_conflict_is_rejected_before_run(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.make_inputs(root)
            out = root / "out"
            out.mkdir()
            args = self.base_args(root)
            normalize(args)
            with self.assertRaisesRegex(FileExistsError, "result already exists"):
                inputs(args)

    def test_output_and_scratch_trees_must_be_disjoint(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.make_inputs(root)
            args = parse_args([
                "--loci-file", str(root / "loci.txt"),
                "--guide-tree", str(root / "guide.nwk"),
                "--out-root", str(root / "out"),
                "--scratch-root", str(root / "out" / "scratch"),
            ])
            with self.assertRaisesRegex(ValueError, "separate directory trees"):
                normalize(args)


if __name__ == "__main__":
    unittest.main()
