"""Native policy composition, sample identities, ranks and analytic boundaries."""
from pathlib import Path
import random
import sys
import tempfile
import unittest

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from iilslib.native_worker import NativeWorker
from iilslib.observations import observe
from iilslib.one_shot_calibration import ThresholdSpec, calibrate
from iilslib.tree_bank import simulate_bank


@unittest.skipUnless((ROOT/'Afresh/bin/iils-statistics').is_file(),'build Afresh first')
class OneShotPolicyTest(unittest.TestCase):
    def test_shared_simulation_and_analytic_strategies_equal_separate_calls(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            guide=root/'guide.nwk'
            guide.write_text("(A,(B,(C,(D,(E,F)'[s1=70;s2=20;s3=10]':0.5)'[s1=60;s2=30;s3=10]':0.3)'[s1=50;s2=25;s3=25]':0.4));\n")
            rng=random.Random(190)
            paths=[]
            for index in range(23):
                path=root/f'{index}.fa'
                path.write_text(''.join(f'>{taxon}\n'+''.join(rng.choices('ACGT',k=240))+'\n'
                                        for taxon in ('ABCDEF' if index<21 else 'ABCDE')))
                paths.append(path)
            binary=ROOT/'Afresh/bin/iils-statistics'
            with NativeWorker(binary,guide,threads=1,seed=0,log=root/'msa.log') as worker:
                observations=observe(paths,guide,worker,transform='raw-alt',batch_size=7,memory_bytes=256<<20)
            bank,_=simulate_bank(root/'bank',guide,observations.taxa,count=40,seed=0,max_cu_length=20)
            options=dict(sample_size=20,sample_seed=0,panel_seed=0,panel_size=8,reservoir=32,
                         binary=binary,guide=guide,statistics_seed=0,threads=2,memory_bytes=256<<20,work=root)
            specs=[ThresholdSpec('low5',policy='standardized-quantile',policy_q='0.05'),
                   ThresholdSpec('median',policy='standardized-quantile',policy_q='0.5'),
                   ThresholdSpec('pooled',policy='standardized-pooled'),
                   ThresholdSpec('analytic-low5',source='analytic-pb',policy='standardized-quantile',policy_q='0.05'),
                   ThresholdSpec('analytic-tail',source='analytic-pb',policy='pooled-tail')]
            shared,receipt=calibrate(observations,specs,bank,**options)
            for spec in specs:
                separate,_=calibrate(observations,[spec],bank if spec.source=='gene-tree-sim' else None,**options)
                for field in ('selected_indices','global_empirical','policy_z'):
                    self.assertEqual(shared[spec.label][field],separate[spec.label][field])
                np.testing.assert_array_equal([r['threshold_score'] for r in shared[spec.label]['rows']],
                                              [r['threshold_score'] for r in separate[spec.label]['rows']])
            self.assertEqual(shared['low5']['global_empirical']['rank'],1)
            self.assertEqual(shared['median']['global_empirical']['rank'],10)
            self.assertEqual(shared['low5']['selected_indices'],shared['median']['selected_indices'])
            self.assertIsNone(shared['analytic-tail']['global_empirical'])
            self.assertTrue(receipt['consumed_holdout'])

    def test_invalid_combinations_fail_before_work(self):
        for spec,transform,size in [
            (ThresholdSpec('bad',source='analytic-pb',policy='standardized-pooled'),'raw-alt',0),
            (ThresholdSpec('bad',policy='standardized-quantile'),'mass',20),
            (ThresholdSpec('bad',q='1'),'raw-alt',0),
            (ThresholdSpec('bad',q='0'),'raw-alt',0),
            (ThresholdSpec('bad'),'raw-alt',20),
        ]:
            with self.assertRaises(ValueError):
                spec.validate(transform,size)
