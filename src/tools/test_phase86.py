"""Metadata identity/grouping regression cases; no data pixels or model imports."""
import unittest
from audit_phase86 import canonical,original_match,components
import importlib.util

class AuditTests(unittest.TestCase):
    def test_old_identity_disambiguates_time_generation_orbit_tile(self):
        old='S2A_OPER_PRD_MSIL1C_PDMC_20160323T192233_R065_V20160323T102143_20160323T102143'
        good='tiles/32/T/MR/S2A_MSIL1C_20160323T102143_N0201_R065_T32TMR_20160323T192233.SAFE/'
        candidates=[good,good.replace('192233.SAFE','192234.SAFE'),good.replace('R065','R066'),good.replace('T32TMR','T32TMS'),good.replace('102143_N','102022_N')]
        self.assertEqual(original_match(old,'32TMR',candidates),[good])
        self.assertEqual(len(original_match(old,'32TMR',[good,good])),2) # ambiguity is not accepted
        self.assertEqual(canonical(good)['sensing_utc'],'2016-03-23T10:21:43Z')

    def test_transitive_not_product_count(self):
        self.assertEqual(components(['a','b','c','d'],[('a','b'),('b','c')]),[['a','b','c'],['d']])
        self.assertEqual(components(['d','c','b','a'],[('c','b'),('b','a')]),[['a','b','c'],['d']])
        self.assertEqual(min(components(['c','b','a'],[('c','b'),('b','a')])[0]),'a')

    def test_bad_canonical_rejected(self):
        for x in ['32TMR','S2A_MSIL2A_20160323T102143_N0201_R065_T32TMR_20160323T192233','random']:
            with self.assertRaises(ValueError):canonical(x)

    @unittest.skipUnless(importlib.util.find_spec('rasterio') and importlib.util.find_spec('shapely'),'existing remote geospatial runtime required')
    def test_cross_projection_and_buffer_contact(self):
        from audit_phase86 import project_geometry,buffered_intersects
        from shapely.geometry import box
        a=box(499000,5000000,500000,5001000)
        def w(g):return project_geometry(g,'EPSG:32632','EPSG:4326')
        # Test inside/outside the fixed2km pair distance, and a touching edge.
        self.assertTrue(buffered_intersects(w(a),w(box(501999.9,5000000,502999.9,5001000)),'EPSG:32632'))
        self.assertFalse(buffered_intersects(w(a),w(box(502000.1,5000000,503000.1,5001000)),'EPSG:32632'))
        self.assertTrue(buffered_intersects(w(a),w(box(500000,5000000,501000,5001000)),'EPSG:32632'))
        z=project_geometry(a,'EPSG:32632','EPSG:32633')
        self.assertTrue(buffered_intersects(w(a),project_geometry(z,'EPSG:32633','EPSG:4326'),'EPSG:32632'))

if __name__=='__main__':unittest.main()
