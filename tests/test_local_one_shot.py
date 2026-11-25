"""Native execution through minimal staging and durable result publication."""
import json
from pathlib import Path
import random
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
from iilslib.local_execution import execute_local
from iilslib.one_shot_results import completed_run
from iilslib.tsv import read_tsv, write_tsv


@unittest.skipUnless((ROOT/'Afresh/bin/iils-statistics').is_file(), 'build Afresh first')
class LocalExecutionTest(unittest.TestCase):
    def test_staged_inputs_publish_durable_paths_and_cleanup(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            guide = root/'guide.nwk'
            guide.write_text("(A,(B,(C,(D,(E,F)'[s1=70;s2=20;s3=10]':0.5)'[s1=60;s2=30;s3=10]':0.3)'[s1=50;s2=25;s3=25]':0.4));\n")
            rng = random.Random(9)
            paths = []
            for index in range(3):
                path = root/f'{index}.fa'
                path.write_text(''.join(f'>{taxon}\n'+''.join(rng.choices('ACGT', k=120))+'\n' for taxon in 'ABCDEF'))
                paths.append(path)
            loci = root/'loci.txt'
            loci.write_text(''.join(str(path)+'\n' for path in paths))
            manifest = root/'inputs.tsv'
            rows = [dict(run_id=run, loci_file=str(loci), guide_tree=str(guide)) for run in ('one', 'two')]
            write_tsv(manifest, list(rows[0]), rows)
            output = root/'published'
            local = root/'node'
            command = [sys.executable, str(ROOT/'scripts/run_manifest_filter_pipeline.py'),
                       '--run-manifest', str(manifest), '--out-root', str(output),
                       '--scratch-root', str(root/'scratch'), '--filter-mode', 'quantile',
                       '--drop-fraction', '0', '--threads', '1', '--seed', '0']
            execute_local(command, local)
            self.assertEqual(list(local.iterdir()), [])
            for run in ('one', 'two'):
                directory = output/run
                receipt = completed_run(directory, run)
                live_status = json.loads((output/'.status'/f'{run}.json').read_text())
                self.assertEqual(live_status['status'], 'completed')
                self.assertEqual(live_status['run_id'], run)
                self.assertEqual(receipt['staging']['copied_files'], 5)
                self.assertEqual((directory/'fixed-fraction/loci.txt').read_text(), loci.read_text())
                result = read_tsv(directory/'results.tsv')[0]
                self.assertEqual(result['decision_tsv'], str(directory/'fixed-fraction/locus_decisions.tsv'))
                self.assertEqual([r['alignment_path'] for r in read_tsv(directory/'inventory.tsv')], list(map(str, paths)))
                self.assertNotIn(str(local), (directory/'configuration.json').read_text())
                self.assertIn('local_execution_receipt.json', receipt['artifacts'])
            with self.assertRaises(FileExistsError):
                execute_local(command, local)

    @unittest.skipUnless((ROOT/'ASTER/bin/caster-site').is_file(), 'build ASTER first')
    def test_staged_name_map_is_used_by_msa_worker_and_filtered_caster(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            guide = root / 'guide.nwk'
            guide.write_text("(A,(B,(C,(D,(E,F)'[s1=70;s2=20;s3=10]':0.5)'[s1=60;s2=30;s3=10]':0.3)'[s1=50;s2=25;s3=25]':0.4));\n")
            name_map = root / 'name_map.tsv'
            name_map.write_text(''.join(f'{taxon}_copy {taxon}\n' for taxon in 'ABCDEF'))
            paths = []
            for index in range(2):
                path = root / f'{index}.fa'
                path.write_text(''.join(f'>{taxon}_copy\n' + 'ACGT' * 40 + '\n' for taxon in 'ABCDEF'))
                paths.append(path)
            loci = root / 'loci.txt'
            loci.write_text(''.join(str(path) + '\n' for path in paths))
            manifest = root / 'inputs.tsv'
            rows = [dict(run_id='mapped', loci_file=str(loci), guide_tree=str(guide), name_map=str(name_map))]
            write_tsv(manifest, list(rows[0]), rows)
            output = root / 'published'
            local = root / 'node'
            command = [sys.executable, str(ROOT / 'scripts/run_manifest_filter_pipeline.py'),
                       '--run-manifest', str(manifest), '--out-root', str(output),
                       '--scratch-root', str(root / 'scratch'), '--filter-mode', 'quantile',
                       '--drop-fraction', '0', '--threads', '1', '--seed', '0',
                       '--build-filtered-tree']
            execute_local(command, local)
            receipt = completed_run(output / 'mapped', 'mapped')
            self.assertEqual(receipt['staging']['copied_files'], 5)
            self.assertIn('name_map_sha256', receipt)
            self.assertTrue((output / 'mapped/fixed-fraction/caster.nwk').is_file())
            self.assertEqual(list(local.iterdir()), [])
