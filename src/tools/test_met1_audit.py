"""Required CPU synthetic cases; no model imports or input imagery."""
import unittest
from run_met1_audit import matrix_stats, independent, summed, difference


class AuditTests(unittest.TestCase):
    def check_both(self, m):
        a, b = matrix_stats(m), independent(m)
        self.assertAlmostEqual(a['old_mIoU'], b['old'], places=15)
        self.assertAlmostEqual(a['reference_classes_mIoU'], b['reference'], places=15)
        self.assertEqual(a['errors'], b['errors'])
        return a

    def test_perfect_single_and_zero_union(self):
        a = self.check_both([[100,0,0],[0,0,0],[0,0,0]])
        self.assertEqual(a['IoU'], [1,None,None])
        self.assertEqual(a['old_mIoU'], 1)
        self.assertEqual(a['U'], a['T'])

    def test_single_false_positive_and_sign(self):
        a = self.check_both([[99,1,0],[0,0,0],[0,0,0]])
        self.assertEqual(a['old_mIoU'], .495)
        self.assertEqual(a['reference_classes_mIoU'], .99)
        self.assertEqual(a['missing_class_false_positives'], {'Cloud':1})
        d = difference(a, matrix_stats([[100,0,0],[0,0,0],[0,0,0]]))
        self.assertEqual(d['error_delta'],1)
        self.assertFalse(d['sign_changed'])

    def test_three_reference_classes(self):
        a = self.check_both([[7,2,1],[1,5,4],[2,1,7]])
        self.assertEqual(a['old_mIoU'], a['reference_classes_mIoU'])
        self.assertEqual(a['surface_cloud_errors'],3)

    def test_reference_class_never_predicted(self):
        a = self.check_both([[8,0,0],[4,0,0],[0,0,2]])
        self.assertEqual(a['IoU'][1],0)
        self.assertEqual(a['p'][1],0)
        self.assertIn(1,a['T'])

    def test_transpose_detection(self):
        m = [[8,2,0],[1,3,1],[0,2,4]]
        reference = [10,5,6]
        matrix_stats(m,reference)
        with self.assertRaises(AssertionError):
            matrix_stats([list(x) for x in zip(*m)],reference)

    def test_pooled_not_macro(self):
        a, b = [[100,0,0],[0,0,0],[0,0,0]], [[0,1,0],[0,0,0],[0,0,0]]
        total = summed([a,b])
        self.assertEqual(total,[[100,1,0],[0,0,0],[0,0,0]])
        s = self.check_both(total)
        self.assertNotEqual(s['old_mIoU'], (matrix_stats(a)['old_mIoU']+matrix_stats(b)['old_mIoU'])/2)

    def test_invalid_matrix(self):
        for m in ([[0,0,0]]*3, [[1.0,0,0],[0,0,0],[0,0,0]], [[-1,0,0],[0,1,0],[0,0,0]]):
            with self.assertRaises(AssertionError): matrix_stats(m)

    def test_sign_flip_with_two_reference_classes(self):
        a = self.check_both([[9,0,1],[0,10,0],[0,0,0]])
        b = self.check_both([[8,2,0],[2,8,0],[0,0,0]])
        d = difference(a,b)
        self.assertTrue(d['sign_changed'])
        self.assertEqual((d['old_sign'],d['reference_sign']),(-1,1))
        self.assertTrue(d['U_count_changed'])


if __name__ == '__main__':
    unittest.main()
