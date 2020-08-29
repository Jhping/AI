#!/usr/bin/python

import unittest
import os,sys
import numpy as np
FNN_path = os.path.dirname(os.path.abspath(__file__))
FNN_path= os.path.dirname(FNN_path)
FNN_path= os.path.dirname(FNN_path)
sys.path.append(FNN_path)
from bigger_than_5_or_not.main import *
from bigger_than_5_or_not.Errors import *

class Test(unittest.TestCase):

    def setUp(self):
        print("setUp")

    def tearDown(self):
        print("tearDown")

    def test_1(self):
        a = 1
        b = 1
        self.assertEqual(a, b)
    def test_Sigmoid(self):
        a = np.array([0,2])
        b = np.array([0.5, 0.88079708])
        self.assertEqual(sigmoid(a).all(), b.all(), "test_Sigmoid_error")

    def test_NoNparrayError(self):
        try:
            raise NoNparrayError("this is not a np array")
        except BaseException as e:
            print(e)

    def test_initialize_parameters(self):
        a,b = initialize_parameters(16)
        self.assertEqual(a.shape, (16,1))
        self.assertEqual(isinstance(b, float) ,True)
        pass
if __name__ == "__main__":
    unittest.main()