import unittest,hashlib
import numpy as np
from phase69_math import box_sum,neighborhood,observable_features,inputs,interval,gate


class Phase69Tests(unittest.TestCase):
    def test_integral_vs_naive_clipped_windows(self):
        a=np.random.default_rng(1).integers(0,2,(6,7))
        for radius in [1,2,15,63]:
            expected=np.array([[a[max(0,y-radius):y+radius+1,max(0,x-radius):x+radius+1].sum() for x in range(7)] for y in range(6)])
            np.testing.assert_array_equal(box_sum(a,radius),expected)

    def test_center_exclusion_and_invalid_pixels(self):
        pred=np.zeros((5,5),np.uint8);valid=np.ones((5,5),bool);pred[2,2]=1
        a=neighborhood(pred,valid,windows=(3,5));self.assertEqual(a[12,0],1);self.assertEqual(a[12,1],0)
        self.assertAlmostEqual(a[6,1],1/8)
        valid[1,1]=False;pred[1,1]=1
        b=neighborhood(pred,valid,windows=(3,5));pred[1,1]=2
        np.testing.assert_array_equal(b,neighborhood(pred,valid,windows=(3,5)))
        valid[:]=False;valid[2,2]=True
        with self.assertRaises(ValueError):neighborhood(pred,valid)

    def test_edge_denominator_and_no_reflection(self):
        p=np.zeros((4,4),np.uint8);v=np.ones((4,4),bool);p[0,1]=1
        x=neighborhood(p,v,windows=(3,5))
        self.assertAlmostEqual(float(x[0,1]),1/3,places=7)
        self.assertAlmostEqual(float(x[0,2]),1/8,places=7)

    def test_joint_permutation_exact_seed_and_center_unchanged(self):
        p=np.random.default_rng(9).integers(0,3,(33,35));v=np.ones(p.shape,bool);v[0,:]=False
        f,a=observable_features(p,v,'product')
        seed=69010+int(hashlib.sha256(b'product').hexdigest()[:8],16)
        perm=np.random.default_rng(seed).permutation(v.sum())
        np.testing.assert_array_equal(f[:,3:],f[perm,1:3]);np.testing.assert_array_equal(f[:,0],(p[v]==1))
        self.assertEqual(a['permutation_sha256'],hashlib.sha256(perm.tobytes()).hexdigest())
        np.testing.assert_array_equal(f,observable_features(p,v,'product')[0])

    def test_four_input_mappings(self):
        old=np.arange(10).reshape(2,5);cloud=np.arange(10,20).reshape(2,5)
        for n,k in [('A',5),('B',6),('D',8),('S',8)]:
            x=inputs(old,cloud,n);self.assertEqual(x.shape,(2,k));np.testing.assert_array_equal(x[:,:5],old)
        np.testing.assert_array_equal(inputs(old,cloud,'S')[:,5:],cloud[:,[0,3,4]])

    def test_descriptive_group_seed(self):
        d=interval([-.1,.1,.2]);self.assertEqual(d,interval([-.1,.1,.2]));self.assertEqual(d['seed'],69010)
        self.assertFalse(d['independent_confirmation'])

    def test_gate_cannot_replace_failed_primary_comparison(self):
        a=dict(macro_recall=.5,miou_percent=78,zero_recall_groups=0)
        d=dict(a,fpr_strata={'all':{'mean':.009}});p={'A':a,'D':d}
        c={n:{'ap':{'ci95':[.001,.02]}} for n in ['D_minus_B','D_minus_S']}
        self.assertTrue(gate(p,c)['all_met']);c['D_minus_S']['ap']['ci95'][0]=0
        self.assertFalse(gate(p,c)['all_met'])


if __name__=='__main__':unittest.main()
