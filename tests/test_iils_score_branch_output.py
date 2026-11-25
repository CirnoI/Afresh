#!/usr/bin/env python3
"""Public contracts for the repository-local iILS statistics interface."""

from __future__ import annotations

import csv
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
    GENE_STATISTICS_FIELDS,
    MSA_ITEM_FIELDS,
    MSA_STATISTICS_FIELDS,
    materialize_msa_scores,
    statistics_binary_default,
    statistics_build_command,
    statistics_command,
)


STATISTICS_BIN = ROOT / "Afresh" / "bin" / "iils-statistics"


class StatisticsInterfaceTest(unittest.TestCase):
    def setUp(self) -> None:
        if not STATISTICS_BIN.is_file():
            self.skipTest(f"statistics interface not built: {STATISTICS_BIN}")

    def test_help_exposes_statistics_only_contract(self) -> None:
        completed = subprocess.run(
            [str(STATISTICS_BIN), "--help"],
            check=True,
            capture_output=True,
            text=True,
        )
        self.assertIn("does not infer a tree or calculate support", completed.stdout)
        self.assertIn("annotations ignored", completed.stdout)
        for forbidden in (
            "--support-source",
            "--support-tree",
            "--score-transform",
            "--branch-output",
            "--threshold",
            "--filter",
        ):
            self.assertNotIn(forbidden, completed.stdout)

    def test_gene_tree_statistics_match_expected_quartet_mass(self) -> None:
        with tempfile.TemporaryDirectory(prefix="iils_gene_statistics_") as tmp:
            tmpdir = Path(tmp)
            species = tmpdir / "species.tre"
            genes = tmpdir / "genes.tre"
            output = tmpdir / "statistics.tsv"
            species.write_text(
                "((A,B)'[s1=9;s2=3;s3=1]',(C,D));\n", encoding="utf-8"
            )
            genes.write_text(
                "((A,B),(C,D));\n"
                "((A,C),(B,D));\n"
                "(A,B,C,D);\n",
                encoding="utf-8",
            )

            subprocess.run(
                [
                    str(STATISTICS_BIN),
                    "--input-type",
                    "gene-tree",
                    "-c",
                    str(species),
                    "-o",
                    str(output),
                    "-t",
                    "1",
                    str(genes),
                ],
                check=True,
                capture_output=True,
                text=True,
            )

            with output.open(newline="", encoding="utf-8") as handle:
                reader = csv.DictReader(handle, delimiter="\t")
                self.assertEqual(reader.fieldnames, GENE_STATISTICS_FIELDS)
                rows = list(reader)
            self.assertEqual([row["gene_id"] for row in rows], ["0", "1", "2"])
            self.assertEqual(
                [float(row["discordance_mass"]) for row in rows],
                [0.0, 1.0, 0.0],
            )

    def test_gene_tree_statistics_accept_leaf_to_species_mapping(self) -> None:
        with tempfile.TemporaryDirectory(prefix="iils_gene_mapping_") as tmp:
            tmpdir = Path(tmp)
            species = tmpdir / "species.tre"
            genes = tmpdir / "genes.tre"
            mapping = tmpdir / "mapping.txt"
            output = tmpdir / "statistics.tsv"
            species.write_text("((A,B),(C,D));\n", encoding="utf-8")
            genes.write_text(
                "((A_copy1:0.1,B_copy1:0.2)95:0.3,"
                "(C_copy1:0.1,D_copy1:0.1)88:0.2);\n"
                "((A_copy2:0.2,C_copy2:0.1)91:0.4,"
                "(B_copy2:0.3,D_copy2:0.2)84:0.2);\n",
                encoding="utf-8",
            )
            mapping.write_text(
                "A_copy1 A\nB_copy1 B\nC_copy1 C\nD_copy1 D\n"
                "A_copy2 A\nB_copy2 B\nC_copy2 C\nD_copy2 D\n",
                encoding="utf-8",
            )

            subprocess.run(
                [
                    str(STATISTICS_BIN),
                    "--input-type",
                    "gene-tree",
                    "-c",
                    str(species),
                    "-a",
                    str(mapping),
                    "-o",
                    str(output),
                    str(genes),
                ],
                check=True,
                capture_output=True,
                text=True,
            )

            with output.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle, delimiter="\t"))
            self.assertEqual([row["gene_id"] for row in rows], ["0", "1"])
            self.assertEqual(
                [float(row["discordance_mass"]) for row in rows],
                [0.0, 1.0],
            )

    def test_msa_statistics_and_item_metadata_have_stable_schemas(self) -> None:
        with tempfile.TemporaryDirectory(prefix="iils_msa_statistics_") as tmp:
            tmpdir = Path(tmp)
            species = tmpdir / "species.tre"
            alignment = tmpdir / "locus.fa"
            loci = tmpdir / "loci.txt"
            statistics = tmpdir / "statistics.tsv"
            items = tmpdir / "items.tsv"
            species.write_text("((A,B),(C,D));\n", encoding="utf-8")
            alignment.write_text(
                ">A\nAAAA\n>B\nAAAA\n>C\nCCCC\n>D\nCCCC\n",
                encoding="utf-8",
            )
            loci.write_text(f"{alignment}\n", encoding="utf-8")

            subprocess.run(
                [
                    str(STATISTICS_BIN),
                    "--input-type",
                    "msa-list",
                    "-c",
                    str(species),
                    "--item-output",
                    str(items),
                    "-o",
                    str(statistics),
                    str(loci),
                ],
                check=True,
                capture_output=True,
                text=True,
            )

            with statistics.open(newline="", encoding="utf-8") as handle:
                reader = csv.DictReader(handle, delimiter="\t")
                self.assertEqual(reader.fieldnames, MSA_STATISTICS_FIELDS)
                branch_rows = list(reader)
            with items.open(newline="", encoding="utf-8") as handle:
                reader = csv.DictReader(handle, delimiter="\t")
                self.assertEqual(reader.fieldnames, MSA_ITEM_FIELDS)
                item_rows = list(reader)
            self.assertEqual(len(branch_rows), 1)
            self.assertEqual(item_rows[0]["n_species_branches"], "1")
            self.assertEqual(item_rows[0]["n_sites_raw"], "4")


    def test_q_zero_branch_does_not_turn_locus_into_preliminary_skip(self) -> None:
        with tempfile.TemporaryDirectory(prefix="iils_msa_materialize_") as tmp:
            tmpdir = Path(tmp)
            statistics = tmpdir / "statistics.tsv"
            items = tmpdir / "items.tsv"
            support = tmpdir / "support.tre"
            branch_output = tmpdir / "branches.tsv"
            score_output = tmpdir / "scores.tsv"
            items.write_text(
                "\t".join(MSA_ITEM_FIELDS)
                + "\n0\tlocus.fa\t4\t10\t10\t1\t2\n",
                encoding="utf-8",
            )
            statistics.write_text(
                "\t".join(MSA_STATISTICS_FIELDS)
                + "\n0\tlocus.fa\t0\t0,1\t0\t0\t0\t0\t0\t10\t4"
                + "\n0\tlocus.fa\t1\t0,1\t9\t3\t1\t13\t1\t10\t4\n",
                encoding="utf-8",
            )
            support.write_text(
                "((A,B)'[s1=9;s2=3;s3=1]',(C,D));\n", encoding="utf-8"
            )

            materialize_msa_scores(
                statistics,
                items,
                support,
                branch_output,
                score_output,
                score_transform="raw-alt",
            )

            with branch_output.open(newline="", encoding="utf-8") as handle:
                self.assertEqual(len(list(csv.DictReader(handle, delimiter="\t"))), 2)
            with score_output.open(newline="", encoding="utf-8") as handle:
                row = list(csv.DictReader(handle, delimiter="\t"))[0]
            self.assertEqual(row["n_valid_branches"], "1")
            self.assertEqual(row["n_skipped_branches"], "1")

    def test_command_and_build_helpers_do_not_target_custom_aster_code(self) -> None:
        self.assertEqual(
            statistics_binary_default(),
            Path("Afresh/bin/iils-statistics"),
        )
        command = [
            str(value)
            for value in statistics_command(
                Path("Afresh/bin/iils-statistics"),
                input_type="gene-tree",
                constraint_tree=Path("tree.tre"),
                input_path=Path("genes.tre"),
                output_path=Path("statistics.tsv"),
                threads=1,
            )
        ]
        self.assertEqual(
            [str(value) for value in statistics_build_command(Path("ASTER"))],
            [
                "make",
                "-C",
                "Afresh",
                "iils-statistics",
                "ASTER_DIR=../ASTER",
            ],
        )
        self.assertNotIn("-C", command)
        for forbidden in (
            "--support-source",
            "--support-tree",
            "--score-transform",
            "--branch-output",
        ):
            self.assertNotIn(forbidden, command)


if __name__ == "__main__":
    unittest.main()
