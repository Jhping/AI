#!/usr/bin/python

import unittest
import sys,os,time
test_path = os.path.abspath(__file__)
test_path = os.path.abspath(test_path)
sys.path.append(test_path)
from bigger_than_5_or_not import main
from TEST.TEST_biger_than_5_or_not import testcase

def suite_TEST_biger_than_5_or_not():
    suite_test = unittest.TestSuite()
    suite_test.addTest(testcase.Test("test_Sigmoid"))
    #suite_test.addTest(testcase.Test("test_2"))
    return suite_test


if __name__ == "__main__":
    #unittest.main(defaultTest="suite")
    runer = unittest.TextTestRunner()
    runer.run(suite_TEST_biger_than_5_or_not())