"""Synthetic checks run before any Phase78 scene pixels."""
import unittest
import numpy as np
from run_phase78 import bilinear,endpoint,validate_directions,evaluate


class Tests(unittest.TestCase):
    def test_gradient(self):
        r,c=np.indices((9,9));x=2*r+3*c
        np.testing.assert_allclose(bilinear(x,np.array([2.25,6.5]),np.array([3.5,1.1])),[15.,16.3])

    def test_edges_no_clamp(self):
        x=np.ones((8,8));r=np.array([-.01,0.,6.9,7.]);c=np.full(4,2.)
        np.testing.assert_array_equal(np.isfinite(bilinear(x,r,c)),[False,True,True,False])

    def test_all_four_even_zero_weight(self):
        x=np.ones((8,8));x[3,3]=np.nan
        self.assertTrue(np.isnan(bilinear(x,np.array([2.]),np.array([2.]))[0]))

    def test_no_data_shared_support(self):
        x=np.ones((8,8));x[4,4]=np.nan
        a=bilinear(x,np.array([2.2,3.2]),np.array([2.2,3.2]));b=np.array([np.nan,1.])
        self.assertFalse((np.isfinite(a)&np.isfinite(b)).any())

    def test_real_geodesics_north_south_utm(self):
        import pyproj
        for crs,aff in [('EPSG:32737',[20,0,451740,0,-20,9058800]),('EPSG:32620',[20,0,432240,0,-20,993900]),('EPSG:32646',[20,0,256600,0,-20,3358880])]:
            to=pyproj.Transformer.from_crs(crs,4326,always_xy=True);back=pyproj.Transformer.from_crs(4326,crs,always_xy=True);g=pyproj.Geod(ellps='WGS84')
            r=np.array([10.,500.,900.]);c=np.array([900.,500.,10.])
            rr,cc,_=endpoint(r,c,110.,0.,aff,to,back,g)
            np.testing.assert_allclose(rr,r,atol=1e-8);np.testing.assert_allclose(cc,c,atol=1e-8)
            self.assertEqual(len(validate_directions(aff,to,back,g,108.1)),40)

    def test_AP_ties_and_missing_classes(self):
        self.assertEqual(evaluate(np.array([0,1,2,2]),np.ones(4))['AP'],.5)
        self.assertEqual(evaluate(np.array([0,1,2,2]),np.array([0.,0.,1.,1.]))['AP'],1.)
        self.assertIsNone(evaluate(np.array([0,1]),np.array([0.,1.]))['AP'])
        self.assertIsNone(evaluate(np.array([2,2]),np.array([0.,1.]))['Shadow_vs_Cloud_AUC'])


if __name__=='__main__':unittest.main()
