"""Verify official class mapping, observation masks and linear RGB policy."""
import unittest
import numpy as np
from prepare_phase65a_explore import convert


class ConversionTest(unittest.TestCase):
    def test_mapping_and_invalid_observations(self):
        raw=np.full((1022,1022,13),.2,dtype=np.float32)
        mask=np.zeros((1022,1022,3),dtype=bool);mask[...,0]=True
        mask[0,0]=[False,True,False];mask[0,1]=[False,False,True]
        mask[0,2]=False;mask[0,3]=True
        raw[0,4,7]=0;raw[0,5,11]=np.nan;raw[0,6,3]=1.2
        rgb,target,valid,features=convert(raw,mask)
        np.testing.assert_array_equal(target[0,:7],[1,2,255,255,255,255,0])
        self.assertTrue(valid[0,2]);self.assertFalse(valid[0,4]);self.assertFalse(valid[0,5])
        self.assertAlmostEqual(float(rgb[0,6,0]),306,places=4)
        self.assertTrue(np.isfinite(rgb).all());self.assertGreater(features['b8_mean'],0)

    def test_wrong_official_dtype_rejected(self):
        raw=np.ones((1022,1022,13),dtype=np.float64)
        mask=np.zeros((1022,1022,3),dtype=bool)
        with self.assertRaises(ValueError):convert(raw,mask)


if __name__=='__main__':unittest.main()
