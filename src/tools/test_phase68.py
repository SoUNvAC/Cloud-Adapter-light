import unittest
import numpy as np
from phase68_math import conditional_weights,dual_threshold,interval,continue_gate
from phase65d_recovery_math import objective,predict_logit,group_threshold


class Phase68Tests(unittest.TestCase):
    def test_conditional_group_class_mass(self):
        ys=[np.array([1,0,0],bool),np.array([1,1,0],bool),np.array([0,0],bool)]
        w=conditional_weights(ys);y=np.concatenate(ys)
        self.assertAlmostEqual(w[y].sum(),.5);self.assertAlmostEqual(w[~y].sum(),.5)
        self.assertAlmostEqual(w[:3][ys[0]].sum(),.25)
        self.assertAlmostEqual(w[3:6][~ys[1]].sum(),1/6)
        with self.assertRaises(ValueError):conditional_weights([np.zeros(3,bool)])

    def test_duplicate_within_group_invariance(self):
        ys=[np.array([1,0],bool),np.array([0,0],bool)]
        values=[np.array([2.,7.]),np.array([3.,5.])]
        original=np.dot(conditional_weights(ys),np.concatenate(values))
        ys[0]=np.repeat(ys[0],3);values[0]=np.repeat(values[0],3)
        self.assertAlmostEqual(original,np.dot(conditional_weights(ys),np.concatenate(values)))

    def test_fixed_scaler_objective_gradient(self):
        x=np.arange(30,dtype=float).reshape(6,5)/7
        model=dict(mean=[1,2,3,4,5],scale=[2,3,4,5,6],theta=[.1,.2,-.2,.3,-.1,.4])
        z=(x-model['mean'])/model['scale'];theta=np.array(model['theta']);y=np.array([1,0,0,1,0,0])
        w=conditional_weights([y[:3],y[3:]])
        loss,grad=objective(theta,z,y,w)
        numeric=[]
        for i in range(6):
            d=np.zeros(6);d[i]=1e-5
            numeric.append((objective(theta+d,z,y,w)[0]-objective(theta-d,z,y,w)[0])/2e-5)
        np.testing.assert_allclose(grad,numeric,atol=1e-9)
        np.testing.assert_array_equal(predict_logit(x,model),(z@theta[:5]+theta[5]).astype(np.float32))

    def test_dual_threshold_ties_and_monotonicity(self):
        arrays=[np.r_[np.ones(2),np.zeros(98)],np.zeros(100)]
        all_t=group_threshold(arrays);dual,parts=dual_threshold(arrays,[True,False])
        self.assertGreaterEqual(dual,all_t)
        for a in arrays:self.assertTrue(np.all((a>=dual)<=(a>=all_t)))
        self.assertLessEqual(np.mean(arrays[0]>=dual),.01)
        with self.assertRaises(ValueError):dual_threshold(arrays,[True,True])

    def test_seed_and_group_interval(self):
        a=interval([0,.1,.2]);self.assertEqual(a,interval([0,.1,.2]))
        self.assertEqual(a['n_groups'],3);self.assertEqual(a['replicates'],10000)
        self.assertFalse(a['independent_confirmation'])

    def test_gate_requires_every_condition(self):
        a=dict(macro_recall=.5,macro_ap=.7,miou_percent=78,zero_recall_groups=0)
        d=dict(a,macro_recall=.6,fpr_strata={'shadow':{'mean':.009},'no_shadow':{'mean':.005}})
        p={'A':a,'B':a,'D':d};self.assertTrue(continue_gate(p)['all_met'])
        d['macro_ap']=.699;self.assertFalse(continue_gate(p)['all_met'])


if __name__=='__main__':unittest.main()
