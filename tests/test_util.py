import os
import sys
import unittest

import numpy as np

sys.path.append(os.path.relpath("../src"))

import utility as ut

class TestUtility(unittest.TestCase):
	def test_util(self):
		a = [1.0, 2.0, 1.0, 2.0]
		b = [2.0, 2.0, 1.0, 4.0]
  
		self.assertEqual(ut.count_lossy_error(a, b, 1), 0.5)
  
		with self.assertRaises(AttributeError) as ctx:
			ut.count_lossy_error([1, 2], [2, 3, 4])

		self.assertTrue("The shapes of prediction and labels should match in dim 0." in str(ctx.exception))
  
		self.assertEquals(ut.count_lossy_error(a, b), 0.25)
  
		self.assertEquals(ut.count_lossy_error(a, b, 4), 0.0)
