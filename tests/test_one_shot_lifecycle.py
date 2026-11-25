"""Fault injection at initialization and publication must not report success."""
import errno
from contextlib import redirect_stderr
import json
import io
from pathlib import Path
import sys
import subprocess
import tempfile
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from iilslib.one_shot import code_identity, execute_run, publish
from iilslib.one_shot_config import parse_args, normalize, inputs
from iilslib.one_shot_results import completed_run


class LifecycleTest(unittest.TestCase):
    def test_unavailable_git_metadata_is_silent(self):
        failure = subprocess.CalledProcessError(128, ['git', 'rev-parse', 'HEAD'])
        with patch('iilslib.one_shot.subprocess.check_output', side_effect=failure):
            stream = io.StringIO()
            with redirect_stderr(stream):
                identity = code_identity()
        self.assertEqual(identity, {
            'commit': None,
            'dirty': None,
            'identity_source': 'git-unavailable',
        })
        self.assertEqual(stream.getvalue(), '')

    def test_publication_copy_failure_releases_only_owned_target(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            final=root/'final'; final.mkdir()
            (final/'scores.tsv').write_text('value\n1\n')
            unrelated=root/'existing'; unrelated.mkdir()
            (unrelated/'keep').write_text('untouched')
            output=root/'published'
            with patch('iilslib.one_shot.shutil.copyfile',side_effect=OSError(errno.ENOSPC,'injected full disk')):
                with self.assertRaises(OSError):
                    publish(final,output,dict(status='completed'))
            self.assertFalse(output.exists())
            self.assertEqual((unrelated/'keep').read_text(),'untouched')
            self.assertEqual(list(root.glob('.iils-publish-*')),[])
            self.assertEqual((final/'scores.tsv').read_text(),'value\n1\n')
            with self.assertRaises(FileExistsError):
                publish(final,unrelated,{})
            self.assertEqual((unrelated/'keep').read_text(),'untouched')

    def test_initialization_error_publishes_failed_receipt_and_cleans_scratch(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            locus=root/'one.fa'; locus.write_text('>A\nACGT\n>B\nACGT\n>C\nACGT\n>D\nACGT\n')
            loci=root/'loci.txt'; loci.write_text(str(locus)+'\n')
            guide=root/'guide.nwk'; guide.write_text("((A,B)'[s1=70;s2=20;s3=10]':0.5,(C,D));\n")
            args=parse_args(['--loci-file',str(loci),'--guide-tree',str(guide),'--out-root',str(root/'out'),
                             '--scratch-root',str(root/'scratch'),'--filter-mode','quantile'])
            specs=normalize(args); run=inputs(args)[0]
            with patch('iilslib.one_shot.code_identity',side_effect=RuntimeError('injected initialization failure')):
                with self.assertRaisesRegex(RuntimeError,'initialization failure'):
                    execute_run(args,run,specs)
            self.assertEqual(list(args.scratch_root.iterdir()),[])
            receipt=json.loads((run.output/'one_shot_receipt.json').read_text())
            self.assertEqual(receipt['status'],'failed')
            status_path=Path(receipt['status_path'])
            self.assertTrue(status_path.is_file())
            status=json.loads(status_path.read_text())
            self.assertEqual(status['status'],'failed')
            self.assertEqual(status['stage'],'initializing')
            with self.assertRaisesRegex(ValueError,'completed'):
                completed_run(run.output,run.run_id)
            self.assertTrue(locus.is_file())
