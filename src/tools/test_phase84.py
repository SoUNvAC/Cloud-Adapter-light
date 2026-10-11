"""Fixed gate boundaries, joint-vector permutation and dense linear readout."""
import unittest
import numpy as np
from run_phase84 import gate,interval,aligned_logits

class Tests(unittest.TestCase):
    def test_fixed_gates(self):
        values=[1.,.5,.5,-1.,True];self.assertTrue(gate(*values)['retain_F_real'])
        for i in range(4):
            x=values.copy();x[i]-=1e-9;self.assertFalse(gate(*x)['retain_F_real'])
        x=values.copy();x[4]=False;self.assertFalse(gate(*x)['retain_F_real'])
    def test_joint_vector_mapping_and_base_unchanged(self):
        f=np.arange(3*4*5,dtype=np.float32).reshape(3,4,5);pos=np.array([1,3,8,11,14,18]);base=np.arange(24,dtype=float).reshape(6,4)
        perm=np.random.default_rng(84011).permutation(6);m=dict(mean=np.arange(7)/3,scale=np.arange(7)+1,theta=list(np.arange(7)/5)+[.4])
        for pp in (None,perm):
            out=aligned_logits(base,f,pos,pp,m);source=pos if pp is None else pos[pp]
            xx=np.column_stack([base,f.reshape(3,-1)[:,source].T]);expected=((xx-m['mean'])/m['scale'])@np.array(m['theta'][:-1])+m['theta'][-1]
            np.testing.assert_allclose(out,expected,rtol=0,atol=1e-12)
        np.testing.assert_array_equal(base,np.arange(24).reshape(6,4))
    def test_group_bootstrap(self):
        x=np.array([-2.,1.,3.]);b=np.random.default_rng(84011).choice(x,(10000,3)).mean(1)
        np.testing.assert_allclose(interval(x)['ci95'],np.percentile(b,[2.5,97.5]))
if __name__=='__main__':unittest.main()
