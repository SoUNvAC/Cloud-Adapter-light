"""Local fixed screening boundaries and hand conditional KL reference."""
import math,unittest
import numpy as np
from phase82_common import continuation_gate,interval

class Tests(unittest.TestCase):
    def test_all_five_gates(self):
        self.assertTrue(continuation_gate(1,-1,-1,-1,.01)['retain_fixed_scheme'])
        base=[1.,-1.,-1.,-1.,.01]
        for i in range(5):
            x=base.copy();x[i]+=1e-8 if i==4 else -1e-8
            self.assertFalse(continuation_gate(*x)['retain_fixed_scheme'])
    def test_conditional_KL_reference(self):
        t=np.array([.8,.2]);s=np.array([.5,.5])
        self.assertAlmostEqual(float(np.sum(t*(np.log(t)-np.log(s)))),.8*math.log(1.6)+.2*math.log(.4))
        self.assertAlmostEqual(float(np.sum(t*(np.log(t)-np.log(t)))),0.)
    def test_group_bootstrap(self):
        x=np.array([-2.,0.,3.]);expected=np.random.default_rng(82011).choice(x,size=(10000,3),replace=True).mean(1)
        np.testing.assert_allclose(interval(x)['ci95'],np.percentile(expected,[2.5,97.5]))

if __name__=='__main__':unittest.main()
