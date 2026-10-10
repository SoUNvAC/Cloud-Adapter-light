"""Statistical contracts: pairing, support, tie AP, one-sided budget inference."""
import unittest
import numpy as np
from phase66_statistics import interval,primary,fpr_constraint
from phase65d_metrics import curve


class Phase66Tests(unittest.TestCase):
    def test_paired_delta_units_and_boundaries(self):
        pairs=[dict(group_id=str(i),delta_ap=v) for i,v in enumerate([.02,.01,0,-.01,-.02])]
        d=primary(pairs)
        self.assertAlmostEqual(d['mean'],0)
        self.assertEqual((d['improved'],d['stable'],d['declined']),(1,3,1))
        self.assertEqual(d['decision'],'increment_not_independently_confirmed')
        with self.assertRaises(ValueError):primary(pairs+[pairs[0]])

    def test_support_and_non_imputation(self):
        self.assertIsNone(interval([])['mean'])
        self.assertIsNone(interval([1])['ci95'])
        pairs=[dict(group_id=str(i),delta_ap=.03) for i in range(15)]
        self.assertEqual(primary(pairs)['decision'],'positive_increment_independently_supported')
        self.assertEqual(primary(pairs,16)['decision'],'insufficient_support')
        self.assertIsNone(curve([.2,.3],[False,False])['ap'])

    def test_one_sided_budget_is_not_two_sided_endpoint(self):
        values=np.asarray([0,.001,.002,.003,.004,.03])
        d=fpr_constraint(values)
        self.assertTrue(d['point_estimate_meets_budget'])
        self.assertFalse(d['upper_bound_meets_budget'])
        self.assertLess(d['upper95_one_sided'],d['ci95'][1])
        self.assertEqual(d['groups_above_budget'],1)
        self.assertEqual(interval(values),interval(values))

    def test_exact_tie_step_ap(self):
        # Single tied threshold precision=.5, recall step=1 -> AP=.5, not PR trapezoid.
        self.assertEqual(curve([.8,.8],[True,False])['ap'],.5)
        self.assertAlmostEqual(curve([.9,.8,.7],[True,False,True])['ap'],5/6)


if __name__=='__main__':unittest.main()
