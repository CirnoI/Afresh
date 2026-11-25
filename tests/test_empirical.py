import sys
import unittest
from decimal import localcontext
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from iilslib.empirical import empirical_rank, empirical_quantile, probability_text


class DecimalEmpiricalTest(unittest.TestCase):
    def test_integer_boundaries_and_real_neighbors(self):
        for q, n, rank in [('0.07',100,7), ('0.28',25,7),
                           ('0.070000000000000000000000000001',100,8),
                           ('0.069999999999999999999999999999',100,7)]:
            self.assertEqual(empirical_rank(q,n),rank)
        self.assertEqual({probability_text(q) for q in ['0.05','0.050','5e-2']},{'0.05'})
        with localcontext() as ctx:
            ctx.prec = 5
            self.assertEqual(probability_text('0.070000000000000000000000000001'),
                             '0.070000000000000000000000000001')

    def test_endpoints_low5_and_median(self):
        for n in range(1,21):
            self.assertEqual(empirical_rank('0',n),1)
            self.assertEqual(empirical_rank('0.05',n),1)
            self.assertEqual(empirical_rank('1',n),n)
        self.assertEqual(empirical_quantile(np.arange(1,21), '0.5').threshold,10)
        r = empirical_quantile([1,2,2,2,3], '0.5')
        self.assertEqual((r.rank,r.ties,r.strict_exceedance_count),(3,3,1))
        # Old lower selects the first of two values; new rank selects the second.
        self.assertEqual(empirical_quantile([1,9],'0.95').threshold,9)

    def test_invalid_input_and_preserved_order(self):
        for q in ['NaN','Infinity','-0.1','1.1','oops']:
            with self.assertRaises(ValueError): empirical_rank(q,20)
        with self.assertRaises(TypeError): empirical_rank(0.07,100)
        for n in [0,-1,1.5,True]:
            with self.assertRaises(ValueError): empirical_rank('0.5',n)
        for samples in [[],[np.nan],[np.inf],[[1,2]]]:
            with self.assertRaises(ValueError): empirical_quantile(samples,'0.5')
        a = np.array([3.,1.,2.])
        empirical_quantile(a,'0.5')
        np.testing.assert_array_equal(a,[3,1,2])


if __name__ == '__main__': unittest.main()
