"""Synthetic checks only: no official image pixels, labels, or remote requests."""
import unittest
import numpy as np
import run_phase76 as m


class MappingChecks(unittest.TestCase):
    def test_d4_unique_and_formulas(self):
        a=np.arange(m.N*m.N).reshape(m.N,m.N)
        views=[np.rot90(a,k) for k in range(4)]
        views += [np.fliplr(x) for x in views[:4]]
        self.assertEqual(len({tuple(x.flat[:3]) for x in views}),8)
        for k,v in enumerate(views):np.testing.assert_array_equal(m.orient(a,k),v)

    def test_fixed_points_and_holdout_not_read_in_selection(self):
        a=np.ones((m.N,m.N)); b=a.copy()
        for i,r in enumerate(m.CENTERS):
            for j,c in enumerate(m.CENTERS):
                if (i+j)%2:b[r-8:r+9,c-8:c+9]=2
        pts,s=m.sample(a,b,'B11','synthetic',('selection',))
        self.assertEqual(len(pts),25);self.assertTrue(s['selection']['pass_gate'])
        self.assertNotIn('heldout',s)
        pts,s=m.sample(a,b,'B11','synthetic')
        self.assertEqual(len(pts),49);self.assertFalse(s['heldout']['pass_gate'])

    def test_support_cannot_hide_invalidity(self):
        a=np.ones((m.N,m.N));b=a.copy();b[:]=np.nan
        pts,s=m.sample(a,b,'B11','invalid')
        self.assertFalse(s['selection']['pass_gate']);self.assertEqual(s['selection']['cube_only_valid'],25*289)

    def test_center_bilinear_full_window_and_nodata(self):
        a=np.arange(160*160,dtype='float64').reshape(160,160)
        a[22,30]=np.nan
        full=m.bilinear_full(a)[10:30,14:34]
        window=m.bilinear_window(a[18:62,26:70],18,26,10,14,size=20)
        np.testing.assert_array_equal(full,window)
        self.assertTrue(np.isnan(full[1,1]))
        self.assertEqual(full[0,0],np.mean(a[20:22,28:30]))

    def test_absolute_affine_for_all_d4(self):
        try:from affine import Affine
        except ImportError:self.skipTest('affine only in remote rasterio environment')
        row={'source_grid':{'resolutions':{'20':dict(XDIM=20,YDIM=-20,ULX=399960,ULY=9100000)},'crs_code':'EPSG:32737'}}
        for k,(a,b,d,e,f,h) in enumerate(m.D4):
            t=Affine(*m.affine_info(row,2060,2589,k)['affine_first6'])
            for r,c in [(0,0),(1021,1021),(128,768)]:
                x,y=t*(c+.5,r+.5)
                self.assertEqual(x,399960+20*(2589+e*r+f*c+h+.5))
                self.assertEqual(y,9100000-20*(2060+a*r+b*c+d+.5))


if __name__=='__main__':unittest.main()
