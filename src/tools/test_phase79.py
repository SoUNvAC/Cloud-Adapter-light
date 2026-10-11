import unittest
import numpy as np
from run_phase79 import auc,darkness_bins,conditional,TASKS


def brute(y,s):
    p=s[y];n=s[~y]
    if not len(p) or not len(n):return None
    return float(np.mean((p[:,None]>n[None,:])+.5*(p[:,None]==n[None,:])))


class Tests(unittest.TestCase):
    def test_auc_ties_half(self):
        self.assertEqual(auc(np.array([1,0,1,0]),np.ones(4)),.5)
        self.assertEqual(auc(np.array([1,0]),np.array([1.,0.])),1.)
        self.assertEqual(auc(np.array([1,0]),np.array([0.,1.])),0.)

    def test_auc_missing_and_empty(self):
        for y in [[],[0,0],[1,1]]:self.assertIsNone(auc(np.array(y),np.ones(len(y))))

    def test_auc_brute_with_ties(self):
        rng=np.random.default_rng(79)
        for _ in range(30):
            y=rng.integers(0,2,30).astype(bool);s=rng.integers(0,5,30)
            self.assertAlmostEqual(auc(y,s),brute(y,s),places=14)

    def test_linear_quantile_duplicate_edges_single_assignment(self):
        dark=np.array([0.,0.,0.,1.,1.,1.,1.,2.,2.,999.]);common=np.arange(10)<9
        edges,ids,original,count=darkness_bins(dark,common)
        np.testing.assert_array_equal(original,np.quantile(dark[common],np.linspace(0,1,21),method='linear'))
        self.assertLess(len(edges),21);self.assertEqual(ids[-1],-1)
        for v in np.unique(dark[common]):self.assertEqual(len(np.unique(ids[common&(dark==v)])),1)
        self.assertEqual(ids[0],0);self.assertEqual(ids[8],count-1)
        np.testing.assert_array_equal(ids[common],np.searchsorted(edges[1:-1],dark[common],side='right'))

    def test_all_equal_single_bin(self):
        edges,ids,q,n=darkness_bins(np.ones(10),np.ones(10,dtype=bool))
        self.assertEqual(n,1);np.testing.assert_array_equal(ids,np.zeros(10));self.assertEqual(edges.tolist(),[1.])

    def test_weighted_within_not_cross_bin(self):
        truth=np.array([2,0,1,2,0]);s=np.array([9.,8.,7.,2.,1.]);ids=np.array([0,0,0,1,1])
        rows,m=conditional(truth,s,ids,2,TASKS[0])
        self.assertEqual(m['conditional_AUC'],1.);self.assertAlmostEqual(m['unstratified_AUC'],4/6)
        self.assertEqual(m['pair_denominator'],3);self.assertEqual([r['pair_weight'] for r in rows],[2,1])

    def test_missing_bin_support_retained_no_merge(self):
        truth=np.array([2,2,0,1,2,0]);s=np.arange(6.);ids=np.array([0,0,1,1,2,2])
        rows,m=conditional(truth,s,ids,3,TASKS[0]);self.assertEqual(len(rows),3)
        self.assertIsNone(rows[0]['AUC']);self.assertIsNone(rows[1]['AUC']);self.assertEqual(m['supported_bins'],1)
        self.assertEqual(m['pair_denominator'],1);self.assertEqual(m['Shadow_coverage'],1/3);self.assertEqual(m['nonShadow_coverage'],1/3)
        _,cloud=conditional(truth,s,ids,3,TASKS[2]);self.assertIsNone(cloud['conditional_AUC'])

    def test_conditional_brute_all_tasks(self):
        rng=np.random.default_rng(79);truth=np.tile([0,1,2],20);ids=rng.integers(0,5,len(truth));s=rng.integers(0,4,len(truth))
        for task in TASKS:
            rows,m=conditional(truth,s,ids,5,task);numerator=0.;den=0
            for b in range(5):
                v=ids==b
                if task!=TASKS[0]:v &= (truth==2)|(truth==(0 if task==TASKS[1] else 1))
                y=truth[v]==2;weight=int(y.sum())*int((~y).sum());a=brute(y,s[v])
                if weight:numerator+=a*weight;den+=weight
            self.assertAlmostEqual(m['conditional_AUC'],numerator/den,places=14)


if __name__=='__main__':unittest.main()
