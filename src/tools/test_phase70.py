import unittest
import numpy as np
from phase70_math import square_sum,interaction,local_contrast,normalized_differences,checked,permutation,interval,promotion
from phase65d_recovery_math import group_threshold


class Tests(unittest.TestCase):
    def test_float_integral_and_clipped_edges(self):
        a=np.arange(30).reshape(5,6)/13
        for radius in [1,30]:
            expected=np.array([[a[max(0,y-radius):y+radius+1,max(0,x-radius):x+radius+1].sum() for x in range(6)] for y in range(5)])
            np.testing.assert_allclose(square_sum(a,radius),expected,atol=1e-13)

    def test_interactions_frozen_scaler_both_sides(self):
        x=np.array([[2.,4.,6.,8.,10.]])
        m=dict(mean=[1,1,1,1,1],scale=[1,3,5,7,9])
        np.testing.assert_array_equal(interaction(x,m),np.ones((1,6)))

    def test_local_valid_neighbors_center_and_no_reflection(self):
        raw=np.ones((5,5,13),np.float32)*.4;valid=np.ones((5,5),bool)
        raw[2,2,[7,11,12]]=.8
        out=local_contrast(raw,valid,3)
        np.testing.assert_allclose(out[12],(.8-.4)/(.4+1e-6),rtol=1e-6)
        valid[1,1]=False;raw[1,1,:]=1000
        before=local_contrast(raw,valid,3);raw[1,1,:]=-1e8
        np.testing.assert_array_equal(before,local_contrast(raw,valid,3))
        valid[:]=False;valid[2,2]=True
        with self.assertRaises(ValueError):local_contrast(raw,valid)

    def test_native_band_indices_guard_and_no_clipping(self):
        raw=np.zeros((1,2,13),np.float32);valid=np.ones((1,2),bool)
        raw[0,0,[2,7,11]]=[.1,.3,.2];raw[0,1,[2,7,11]]=[1,-1,2]
        f,a=normalized_differences(raw,valid)
        np.testing.assert_allclose(f[0],[-.5,.2,-1/3],atol=1e-7)
        self.assertEqual(f[1,0],2e6);self.assertEqual(a['protected_count'],[1,0,0])
        self.assertEqual(a['protected_fraction'][0],.5)

    def test_product_branch_joint_shuffle(self):
        a=np.arange(60).reshape(10,6);p,d=permutation('id','A',10);other,_=permutation('id','B',10)
        np.testing.assert_array_equal(p,permutation('id','A',10)[0]);self.assertFalse(np.array_equal(p,other))
        np.testing.assert_array_equal(a[p,1]-a[p,0],np.ones(10))

    def test_nonfinite_and_no_clipping(self):
        self.assertEqual(checked([2e6])[0],2e6)
        with self.assertRaises(ValueError):checked([np.nan])
        with self.assertRaises(ValueError):checked([1e100])

    def test_threshold_ties_budget(self):
        a=[np.ones(100),np.zeros(50)];t=group_threshold(a,.01)
        self.assertLessEqual(np.mean([(x>=t).mean() for x in a]),.01)

    def test_bootstrap_group_seed(self):
        x=interval([-.1,.2,.1]);self.assertEqual(x,interval([-.1,.2,.1]));self.assertEqual(x['seed'],70010)

    def test_promotion_requires_four_strict_gt_1pp_and_all_conditions(self):
        base=dict(macro_ap=.5,macro_recall=.5,miou_percent=70,zero_recall_groups=0)
        p={'L3':base};c={};status={n:{'status':'completed'} for n in 'ABC'}
        for n in 'ABC':
            p[n]=dict(base,macro_ap=.52,fpr_strata={'all':{'mean':.009}});p[n+'_shuffle']=dict(base)
            c[n+'_minus_L3']=dict(ap={'mean':.02},pairs=[{'delta_ap':.02} for _ in range(7)])
        self.assertEqual(promotion(p,c,status)['selected'],['B','C'])
        c['B_minus_L3']['pairs']=[{'delta_ap':v} for v in [.01,.01,.01,.01,.01,.01,.08]]
        self.assertFalse(promotion(p,c,status)['branches']['B']['eligible'])


if __name__=='__main__':unittest.main()
