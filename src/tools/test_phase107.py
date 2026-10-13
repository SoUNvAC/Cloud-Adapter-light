"""Synthetic AP/NA/group-bootstrap checks, no fits or research pixels."""
import unittest
from phase107_common import ap,comparisons

class Contracts(unittest.TestCase):
    def test_ties_and_NA(self):
        self.assertAlmostEqual(ap([1,0,1,0],[.9,.9,.8,.2]),7/12)
        self.assertIsNone(ap([0,0],[.4,.2]))
    def test_fixed_comparisons(self):
        rows=[dict(tile=str(i),A_AP=.3,R_AP=.4,S_AP=.5) for i in range(8)]
        out=comparisons(rows,'PixBox')
        self.assertEqual([r['comparison'] for r in out],['S-R','R-A'])
        self.assertTrue(out[0]['primary']);self.assertFalse(out[1]['primary'])
        self.assertAlmostEqual(out[0]['mean_delta_AP_pp'],10)
        self.assertEqual(out[0]['improved'],8)

if __name__=='__main__':unittest.main()
