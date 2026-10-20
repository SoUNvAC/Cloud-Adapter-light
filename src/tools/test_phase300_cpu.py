"""Synthetic tests only; no experiment data, predictions or labels loaded."""
import unittest
import numpy as np
import analyze_phase300 as analysis


class FixedRules(unittest.TestCase):
    def test_absent_union_excluded(self):
        result = analysis.ious([[4,0,0],[0,0,0],[0,0,2]])
        self.assertTrue(np.isnan(result[1]))
        self.assertEqual(float(np.nanmean(result)),1.)

    def test_oracle_ties_and_global_budget(self):
        groups = {
            'a':dict(y=np.array([2,0,0]),scores={'R4':np.array([.8,.8,.2])}),
            'b':dict(y=np.array([2,0]),scores={'R4':np.array([.7,.8])})}
        summary = [dict(strategy=name,macro_negative_FPR_pp=budget,
                        macro_Shadow_recall_pp=100.,macro_Shadow_AP_pp=50.)
                   for name,budget in [('R4',0.),('B-PL',25.),('M-PL',75.)]]
        rows = analysis.oracle('synthetic_only',groups,summary)
        self.assertEqual(rows[0]['R4_actual_FPR_pp'],0.)
        self.assertEqual(rows[0]['R4_oracle_recall_pp'],0.)
        self.assertGreater(rows[0]['R4_oracle_global_threshold'],.8)
        self.assertEqual(rows[1]['R4_actual_FPR_pp'],75.)
        self.assertEqual(rows[1]['R4_oracle_recall_pp'],100.)
        self.assertEqual(rows[1]['R4_oracle_global_threshold'],.7)


if __name__ == '__main__':
    unittest.main()
