import unittest
import numpy as np
from phase65d_metrics import curve,threshold_at_budget,corrected,confusion,measures


class RankingTests(unittest.TestCase):
    def test_all_ties_are_not_split(self):
        c=curve([.3,.3,.3,.3],[True,False,True,False])
        self.assertAlmostEqual(c['ap'],.5);self.assertAlmostEqual(c['roc_auc'],.5)
        self.assertEqual(c['matched_fpr']['0.01']['recall'],0)

    def test_perfect_ranking_and_monotone_transform(self):
        x=np.array([.9,.8,.2,.1]);y=[True,True,False,False]
        a=curve(x,y);b=curve(x**3,y)
        self.assertEqual(a['ap'],1);self.assertEqual(a['roc_auc'],1)
        self.assertEqual(a['ap'],b['ap']);self.assertEqual(a['matched_fpr']['0.001']['recall'],1)

    def test_threshold_budget_conservative_with_ties(self):
        s=np.array([.1,.2,.2,.8],dtype=np.float32)
        t=threshold_at_budget(s,2)
        self.assertLessEqual(int((s>=t).sum()),2);self.assertEqual(int((s>=t).sum()),1)
        self.assertEqual(int((s>=threshold_at_budget(s,0)).sum()),0)

    def test_ignore_label_and_nonshadow_fallback(self):
        pred=corrected(np.array([.1,.9,.8]),.5,np.array([1,0,0]))
        cm=confusion(np.array([0,2,255],dtype=np.uint8),pred)
        m=measures(cm);self.assertEqual(m['recall'],1);self.assertEqual(m['false_positive'],0)
        self.assertEqual(int(cm.sum()),2)

    def test_no_positive_support_is_explicit(self):
        self.assertIsNone(curve([.1,.2],[False,False])['ap'])


if __name__=='__main__':unittest.main()
