"""Small contracts exercise real split and exact-tie calibration, no model fitting."""
import hashlib, unittest
import numpy as np
from phase65d_recovery_math import group_threshold
class Contract(unittest.TestCase):
    def test_split_implementation(self):
        # AST isolates the actual split function without remote runtime imports.
        import ast
        from pathlib import Path
        tree=ast.parse(Path(__file__).with_name('run_phase94.py').read_text(encoding='utf-8-sig'))
        fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='split_rows')
        scope={'hashlib':hashlib};exec(compile(ast.Module(body=[fn],type_ignores=[]),'<split>','exec'),scope)
        rows=[dict(group_id=f'group_{i}',product=f'product_{i}') for i in range(153)]
        a,b=scope['split_rows'](rows);c,d=scope['split_rows'](list(reversed(rows)))
        self.assertEqual((a,b),(c,d));self.assertEqual((len(a),len(b)),(100,53))
        self.assertFalse({r['group_id'] for r in a}&{r['group_id'] for r in b})
        with self.assertRaises(AssertionError):scope['split_rows'](rows[:-1]+[rows[0]])
    def test_exact_tie_boundary(self):
        arrays=[np.array([.9,.8,.8,.2],np.float32),np.array([.8,.4],np.float32)]
        t=group_threshold(arrays,.2);self.assertEqual(t,float(np.float32(.9)))
        self.assertLessEqual(np.mean([(a>=t).mean() for a in arrays]),.2)
        self.assertGreater(np.mean([(a>=np.float32(.8)).mean() for a in arrays]),.2)
    def test_no_admissible_observed_threshold(self):
        arrays=[np.array([.5,.5],np.float32)]
        t=group_threshold(arrays,.01);self.assertGreater(t,.5)
        self.assertEqual(np.count_nonzero(arrays[0]>=t),0)
if __name__=='__main__':unittest.main()
