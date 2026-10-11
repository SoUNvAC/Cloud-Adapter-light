import unittest
import numpy as np
from phase85_common import gate,interval
from phase65d_recovery_math import group_threshold
class Checks(unittest.TestCase):
 def test_paired_seed_mean_is_seven_groups(self):
  a=np.array([1,3,5,-2,4,7,2]);b=np.array([2,-1,3,6,1,4,8]);d=(a+b)/2
  rng=np.random.default_rng(85011);idx=rng.integers(0,7,(10000,7))
  bs=((a[idx]+b[idx])/2).mean(1)
  np.testing.assert_array_equal(interval(d)['ci95'],np.quantile(bs,[.025,.975]));self.assertEqual(interval(d)['n_groups'],7)
 def test_fixed_gate_boundaries(self):
  def p(ap=.7,rec=.7,fpr=.01,macro=80,pool=80):return dict(macro_ap={'mean':ap},macro_recall={'mean':rec},macro_fpr={'mean':fpr},macro_miou_percent={'mean':macro},miou_percent=pool)
  rows={'MM':p(),'LL66':p(.72),'LL67':p(.72),'LM66':p(),'LM67':p()};self.assertTrue(all(gate(rows).values()))
  rows['LL66']=p(.7);self.assertFalse(gate(rows)['both_AP_gains_positive'])
  rows['LL66']=p(.72,fpr=.01001);self.assertFalse(gate(rows)['each_macro_FPR_at_most_1percent'])
  rows['LM67']=p(pool=79.99);self.assertFalse(gate(rows)['each_LM_pooled_mIoU_at_least_MM'])
 def test_exact_ties(self):
  neg=[np.r_[np.full(1,3,np.float32),np.zeros(99,np.float32)],np.zeros(100,np.float32)]
  t=group_threshold(neg,.01);self.assertEqual(t,3);self.assertLessEqual(np.mean([(a>=t).mean() for a in neg]),.01)
if __name__=='__main__':unittest.main()
