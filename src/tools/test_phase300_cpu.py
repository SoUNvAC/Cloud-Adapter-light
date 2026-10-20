"""Synthetic tests only; no experiment data, predictions or labels loaded."""
import unittest
import numpy as np
import analyze_phase300 as analysis


class FixedRules(unittest.TestCase):
    def test_absent_union_excluded(self):
        result = analysis.ious([[4,0,0],[0,0,0],[0,0,2]])
        self.assertTrue(np.isnan(result[1]))
        self.assertEqual(float(np.nanmean(result)),1.)

    def test_group_mean_not_pixel_weighted(self):
        large = np.array([[1000,0,0],[0,0,0],[0,0,0]])
        small = np.array([[0,1,0],[0,0,0],[0,0,0]])
        group_mean = np.mean([np.nanmean(analysis.ious(m)) for m in (large,small)])
        pooled = np.nanmean(analysis.ious(large+small))
        self.assertEqual(group_mean,.5)
        self.assertNotEqual(group_mean,pooled)

    def test_ap_ties_and_missing_positives(self):
        result,_ = analysis.reference.metric(np.array([2,0,0],np.uint8),
             np.array([2,2,0],np.uint8),np.array([.8,.8,.1],np.float32))
        self.assertEqual(result['AP'],.5)
        missing,_ = analysis.reference.metric(np.array([0,1],np.uint8),
             np.array([0,1],np.uint8),np.array([.1,.2],np.float32))
        self.assertIsNone(missing['AP'])
        self.assertIsNone(missing['recall'])

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
