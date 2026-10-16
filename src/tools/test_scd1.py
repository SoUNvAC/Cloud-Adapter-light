"""In-memory synthetic SCD geometry and metric checks only."""
import unittest
from scd1_common import dedup,geometry,memberships,metrics


class SCDTests(unittest.TestCase):
    def test_half_open_and_pixel_centres(self):
        patches=[(0,2,0,2),(0,2,2,4)]
        self.assertEqual(geometry(0,1,patches,20),(1,10))
        self.assertEqual(geometry(0,2,patches,20),(1,10))
        with self.assertRaises(AssertionError):geometry(2,1,patches,20)
    def test_duplicate_patches(self):
        p=dedup([[0,4,0,4],[0,4,0,4],[0,4,2,6]])
        self.assertEqual(len(p),2)
        self.assertEqual(geometry(1,2,p,20),(2,30))
    def test_four_subcells(self):
        self.assertEqual({str(r%2)+str(c%2) for r in (2,3) for c in (4,5)},{'00','01','10','11'})
    def test_no_coverage_stops(self):
        with self.assertRaises(AssertionError):geometry(9,9,[(0,2,0,2)],10)
    def test_fixed_strata_threshold(self):
        r=dict(subcell='11',C20_coverage_count=3,C20_best_margin_m=300)
        self.assertEqual(memberships(r),('all','subcell 11','C20 coverage>=3','C20 margin>=300m'))
    def test_missing_classes(self):
        self.assertEqual(metrics([0,0],[.2,.8])['AP_NA_reason'],'no_positive')
        self.assertEqual(metrics([1,1],[.2,.8])['AP_NA_reason'],'no_negative')
        self.assertIsNone(metrics([],[])['AP'])
        self.assertIsNone(metrics([1,1],[.2,.8])['FPR'])
    def test_tied_AP(self):
        s=metrics([1,0,1,0],[.9,.9,.8,.2])
        self.assertAlmostEqual(s['AP'],7/12,places=14)
        self.assertEqual((s['TP'],s['FP']),(2,1))
    def test_empty_layer_retained(self):
        self.assertTrue(metrics([],[])['sparse'])
        self.assertEqual(metrics([],[])['positive'],0)


if __name__=='__main__':unittest.main()
