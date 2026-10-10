import unittest
import numpy as np
from phase71_math import rgb_local,inputs
from phase70_math import local_contrast


class Tests(unittest.TestCase):
    def test_same_formula_and_native_rgb_channels(self):
        rng=np.random.default_rng(71010);raw=rng.uniform(.01,.8,(12,9,13)).astype('float32');valid=rng.uniform(size=(12,9))>.1
        expected=raw.copy();expected[...,[7,11,12]]=raw[...,[3,2,1]]
        np.testing.assert_allclose(rgb_local(raw,valid),local_contrast(expected,valid).astype('float32'),rtol=0,atol=0)
    def test_center_excluded_and_invalid_neighbors_ignored(self):
        raw=np.full((3,3,13),.4,'float32');v=np.ones((3,3),bool);raw[1,1,[3,2,1]]=.8
        np.testing.assert_allclose(rgb_local(raw,v)[4],(.8-.4)/(.4+1e-6),rtol=1e-6)
        v[0,0]=False;a=rgb_local(raw,v);raw[0,0]=np.nan;np.testing.assert_array_equal(a,rgb_local(raw,v))
        v[:]=False;v[1,1]=True
        with self.assertRaises(ValueError):rgb_local(raw,v)
    def test_drop_exact_local_column_keep_original_five(self):
        x=np.arange(20).reshape(4,5);b=np.arange(12).reshape(4,3);rgb=b+100
        for name,cols in [('B_noB08',[1,2]),('B_noB11',[0,2]),('B_noB12',[0,1])]:
            out=inputs(x,b,rgb,name);np.testing.assert_array_equal(out[:,:5],x);np.testing.assert_array_equal(out[:,5:],b[:,cols]);self.assertEqual(out.shape[1],7)
        np.testing.assert_array_equal(inputs(x,b,rgb,'RGB_local')[:,5:],rgb)
        self.assertEqual(inputs(x,b,rgb,'B').shape[1],8)


if __name__=='__main__':unittest.main()
