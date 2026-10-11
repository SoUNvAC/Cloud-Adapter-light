import unittest
import numpy as np
from phase89_math import area, all_valid, sigmoid, calibrate, decisions, interval

class Contracts(unittest.TestCase):
    def test_area_and_joint_support(self):
        x=np.arange(36).reshape(6,6)
        np.testing.assert_allclose(area(x,2),[[3.5,5.5,7.5],[15.5,17.5,19.5],[27.5,29.5,31.5]])
        v=np.ones((6,6),bool);v[0,0]=False
        self.assertEqual(int(all_valid(v,2).sum()),8)
        self.assertFalse(all_valid(all_valid(v,2),3)[0,0])
        np.testing.assert_allclose(area(np.ones((6,6,3))*.3,3),.3)

    def test_edge_padding_excluded(self):
        x=np.zeros((1020,1020));x[:21,:9]=4
        a=area(x[:21,:9],3)
        self.assertEqual(a.shape,(7,3));np.testing.assert_allclose(a,4)

    def test_class_and_frozen_probability_threshold(self):
        m={'msre_source':dict(threshold=-2.),'msre_source_spectral':dict(threshold=-2.)}
        p={'Source':np.array([[[.7,.2,.1]]]),'MsRE':np.array([[[.2,.7,.1]]]),
           'L2':np.array([[sigmoid(-2.)]]),'L3':np.array([[sigmoid(-3.)]])}
        d=decisions(p,m)
        self.assertEqual([d[n].item() for n in ['Source','MsRE','L2','L3']],[0,1,2,1])

    def test_calibrate_before_aggregate(self):
        model=dict(mean=[0,0],scale=[1,1],theta=[1,0,1],threshold=-2.)
        model3=dict(mean=[0]*5,scale=[1]*5,theta=[1,0,0,0,0,1],threshold=-2.)
        ms=np.array([[.01,.01,.99]]*3);ss=np.full((3,3),.4);sp=np.ones((3,3,3))
        p=calibrate(ms,ss,sp,dict(msre_source=model,msre_source_spectral=model3))
        self.assertGreater(abs(area(p['L2'],3).item()-sigmoid(np.log(ms.mean()/(1-ms.mean()))+1)),.05)
        np.testing.assert_allclose(p['L2'],p['L3'])

    def test_bootstrap_groups_not_pixels(self):
        a=interval([1,2,3]);b=interval([1,2,3]);self.assertEqual(a,b)
        self.assertEqual(a['n_groups'],3);self.assertEqual(a['mean'],2)
        self.assertIsNone(interval([])['ci95'])

if __name__=='__main__':unittest.main()
