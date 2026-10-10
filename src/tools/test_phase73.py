"""Fixed sampling, native-window edge padding and diagnostic selection invariants."""
import unittest
import numpy as np
from run_phase73 import sample,crop,choose

class FixedDiagnosis(unittest.TestCase):
    def test_sampling(self):
        mask=np.ones((8,9),bool)
        a,record=sample(mask,'product',0);b,again=sample(mask,'product',0)
        np.testing.assert_array_equal(a,b)
        self.assertEqual(record,again);self.assertEqual(len(set(a)),4)
        empty,rec=sample(np.zeros((2,2),bool),'product',0)
        self.assertEqual(len(empty),0);self.assertEqual(rec['candidates'],0)
        all_points,_=sample(np.eye(3,dtype=bool),'product',1)
        np.testing.assert_array_equal(all_points,[0,4,8])
    def test_window(self):
        a=np.arange(20,dtype=np.int32).reshape(4,5)
        out=crop(a,0,0,-1)
        self.assertEqual(out.shape,(257,257));self.assertEqual(out[128,128],0)
        np.testing.assert_array_equal(out[128:132,128:133],a)
        self.assertEqual(out[127,128],-1)
    def test_ties_undefined_and_mandatory(self):
        rows=[dict(product=p,delta_fpr=1.,delta_ap_pp=None if p=='z' else 0.) for p in ['z','c','b','a','x_T06CWV_y']]
        picks=choose(rows)
        self.assertIn('x_T06CWV_y',picks);self.assertNotIn('z',picks)
        self.assertEqual(picks['a']['reasons'],['worst_fpr','worst_ap','best_ap'])

if __name__=='__main__':unittest.main()
