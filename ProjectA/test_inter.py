import unittest
import linear


class LinearTest(unittest.TestCase):
    def setUp(self):
        self.xs = [0, 1, 2, 3, 4, 5, 6, 7]
        self.ys = [0, 1, 2, 3, 4, 5, 6, 7]

    def test_trivial_case(self):
        self.assertEqual(linear.linint(0.2, self.xs, self.ys), 0.2)

    def test_out_of_range_error_above(self):
        with self.assertRaises(ValueError):
            linear.linint(15, self.xs, self.ys)

    def test_out_of_range_error_below(self):
        with self.assertRaises(ValueError):
            linear.linint(-1, self.xs, self.ys)
