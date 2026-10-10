import unittest
import numpy as np
from phase67_controls import combined,selected,choose,CONTROLS


class ControlsTests(unittest.TestCase):
    def test_exact_ablation_and_dimension_matching(self):
        x=combined(np.array([[10,20,8,11,12]]),np.array([[4,3,2]]))
        np.testing.assert_array_equal(selected(x,'L2_RGB'),[[10,20,4,3,2]])
        np.testing.assert_array_equal(selected(x,'Source_spectral'),[[20,8,11,12]])
        changed=x.copy();changed[:,0]=99999
        np.testing.assert_array_equal(selected(changed,'Source_spectral'),selected(x,'Source_spectral'))
        np.testing.assert_array_equal(selected(x,'MsRE_spectral'),[[10,8,11,12]])
        self.assertEqual(CONTROLS['Source_spectral']['nonshadow'],'source')
        self.assertEqual(CONTROLS['L2_RGB']['positive_features'],2)
        self.assertEqual(CONTROLS['Source_spectral']['positive_features'],1)
        with self.assertRaises(ValueError):selected(x,'extra_feature')
        with self.assertRaises(ValueError):combined(np.zeros((2,5)),np.zeros((1,3)))

    def test_selection_budget_before_ap_and_no_budget_relaxation(self):
        rows={n:dict(macro_ap=.5,macro_fpr=.005,n_features=4 if n.endswith('spectral') else 5) for n in [*CONTROLS,'L3']}
        rows['L2_RGB']['macro_ap']=.99;rows['L2_RGB']['macro_fpr']=.01001
        self.assertEqual(choose(rows),'MsRE_spectral')
        rows['L3']['macro_ap']=.6
        self.assertEqual(choose(rows),'L3')
        for r in rows.values():r['macro_fpr']=.02
        self.assertIsNone(choose(rows))


if __name__=='__main__':unittest.main()
