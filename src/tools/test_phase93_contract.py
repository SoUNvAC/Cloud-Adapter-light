"""No test estimator fits: preserve the exactly-two-fitting contract."""
import unittest
from types import SimpleNamespace
import numpy as np
from phase93_math import selected, equal_weights, independent_probability, PARAMS
from phase65d_recovery_math import group_threshold
from phase89_math import area


class Contracts(unittest.TestCase):
    def test_order_and_weight_scale(self):
        x = np.arange(24).reshape(3, 8)
        np.testing.assert_array_equal(selected(x, 'G6')[0], [5,6,7,2,3,4])
        np.testing.assert_array_equal(selected(x, 'G7')[0], [5,6,7,2,3,4,1])
        w = equal_weights([2,4,6])
        self.assertAlmostEqual(w.sum(), 12)
        self.assertAlmostEqual(w[:2].sum(), w[2:6].sum())
        self.assertAlmostEqual(w[:2].sum(), w[6:].sum())

    def test_fixed_parameters(self):
        self.assertEqual(PARAMS['random_state'],93012)
        self.assertEqual(PARAMS['max_iter'],200)
        self.assertFalse(PARAMS['early_stopping'])
        self.assertIsNone(PARAMS['class_weight'])
        self.assertEqual(PARAMS['max_features'],1.)

    def test_probability_threshold_no_second_sigmoid(self):
        negatives=[np.array([.7,.8,.85,.9],np.float32)]
        t=group_threshold(negatives,.25)
        self.assertLessEqual(np.mean(negatives[0]>=t),.25)
        self.assertEqual(int(np.sum(negatives[0]>=t)),1)
        self.assertNotEqual(int(np.sum(negatives[0]>=1/(1+np.exp(-t)))),1)

    def test_independent_numeric_tree_and_probability_area(self):
        dtype=[('is_leaf','u1'),('is_categorical','u1'),('feature_idx','i8'),
               ('num_threshold','f8'),('left','i8'),('right','i8'),('value','f8')]
        nodes=np.zeros(3,dtype=dtype)
        nodes[0]['num_threshold']=.5;nodes[0]['left']=1;nodes[0]['right']=2
        nodes[1:]['is_leaf']=1;nodes[1]['value']=-2;nodes[2]['value']=2
        m=SimpleNamespace(_baseline_prediction=np.array([[0.]]),
                          _predictors=[[SimpleNamespace(nodes=nodes)]])
        x=np.array([[.5],[.6]])
        result=independent_probability(m,x)
        np.testing.assert_array_equal(result,(1/(1+np.exp(-np.array([-2.,2.])))).astype(np.float32))
        pixels=np.array([[0.],[0.],[1.]]*3)
        averaged=area(independent_probability(m,pixels).reshape(3,3).astype(np.float64),3)
        wrong=independent_probability(m,np.array([[pixels.mean()]]))[0]
        self.assertNotAlmostEqual(float(averaged[0,0]),float(wrong),places=3)


if __name__=='__main__':unittest.main()
