"""Gradient/checkerboard/NoData tests with no real dataset reads."""
import unittest
import numpy as np
import run_phase77 as m


class ResamplingChecks(unittest.TestCase):
    def test_twelve_candidates_ten_classes(self):
        self.assertEqual(len(m.CANDIDATES),12)
        self.assertEqual(len({c['equivalence'] for c in m.CANDIDATES}),10)

    def test_half_pixel_scopes_identical_absolute_coordinates(self):
        r0=2060;base=r0-65
        tile=m.coordinates(10980,5490,r0,1022,'half_pixel')
        outer=2*base+m.coordinates(2304,1152,65,1022,'half_pixel')
        cube=2*r0+m.coordinates(2044,1022,0,1022,'half_pixel')
        np.testing.assert_array_equal(tile,outer);np.testing.assert_array_equal(tile,cube)

    def test_linear_gradient_both_centers(self):
        r,c=np.indices((40,50));raw=3*r+5*c+7
        for center in ['half_pixel','align_corners']:
            rr=m.coordinates(40,20,0,20,center);cc=m.coordinates(50,25,0,25,center)
            np.testing.assert_allclose(m.bilinear(raw,rr,cc),3*rr[:,None]+5*cc[None,:]+7,atol=1e-12)
        self.assertEqual(m.coordinates(40,20,0,20,'align_corners')[-1],39)
        self.assertEqual(m.coordinates(40,20,0,20,'half_pixel')[-1],38.5)

    def test_checkerboard_interpolation_and_integer_controls(self):
        r,c=np.indices((40,40));raw=((r+c)%2).astype(float)
        target=2*np.arange(4,14)
        np.testing.assert_array_equal(m.bilinear(raw,target+.5,target+.5),np.full((10,10),.5))
        for dr in [0,1]:
            for dc in [0,1]:np.testing.assert_array_equal(m.gather(raw,target+dr,target+dc),np.full((10,10),(dr+dc)%2))

    def test_gaussian_fusion_matches_independent_two_stage(self):
        raw=np.random.default_rng(77).normal(size=(40,40));g=np.exp(-.5*(np.arange(-2,3)/.5)**2);g/=g.sum()
        filtered=np.full_like(raw,np.nan)
        for r in range(2,38):
            for c in range(2,38):filtered[r,c]=np.sum(raw[r-2:r+3,c-2:c+3]*np.outer(g,g))
        target=2*np.arange(3,15);offsets,weights=m.gaussian_kernel()
        fused=m.separable(raw,target,target,offsets,weights)
        reference=m.bilinear(filtered,target+.5,target+.5)
        np.testing.assert_allclose(fused,reference,atol=3e-16)

    def test_filters_preserve_gradient_and_triangle_checkerboard(self):
        r,c=np.indices((40,40));raw=(3*r+5*c+7).astype(float);target=2*np.arange(3,15)
        for offsets,weights in [m.gaussian_kernel(),([-1,0,1,2],np.array([1,3,3,1])/8)]:
            np.testing.assert_allclose(m.separable(raw,target,target,offsets,weights),3*(target[:,None]+.5)+5*(target[None,:]+.5)+7,atol=1e-12)
        checker=((r+c)%2).astype(float)
        np.testing.assert_array_equal(m.separable(checker,target,target,[-1,0,1,2],np.array([1,3,3,1])/8),np.full((12,12),.5))

    def test_nodata_and_no_reflection(self):
        raw=np.ones((40,40));raw[10,10]=np.nan
        self.assertTrue(np.isnan(m.bilinear(raw,np.array([10.5]),np.array([10.5]))[0,0]))
        # A zero-weight neighboring NoData is not an actual participant.
        self.assertEqual(m.bilinear(raw,np.array([9.]),np.array([9.]))[0,0],1)
        for offsets,weights in [m.gaussian_kernel(),([-1,0,1,2],np.array([1,3,3,1])/8)]:
            self.assertTrue(np.isnan(m.separable(raw,np.array([0]),np.array([0]),offsets,weights)[0,0]))
            self.assertTrue(np.isnan(m.separable(raw,np.array([10]),np.array([10]),offsets,weights)[0,0]))


if __name__=='__main__':unittest.main()
