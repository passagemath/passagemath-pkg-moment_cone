import unittest

from cone.partition import Partition

class TestPartition(unittest.TestCase):

    def test_interface(self):
        p = Partition([4, 4, 3, 2, 1, 0, 0])
        self.assertEqual(len(p), 5)
        self.assertEqual(p[0], 4)
        self.assertEqual(p[3], 2)
        self.assertEqual(p[10], 0)
        self.assertEqual(list(p), [4, 4, 3, 2, 1])
        self.assertEqual(p.pad(6), (4, 4, 3, 2, 1, 0))

    def test_non_decreasing(self):
        with self.assertRaises(Exception):
            p = Partition([4, 4, 3, 2, 3, 0])

    def test_negative(self):
        with self.assertRaises(Exception):
            p = Partition([4, 4, 3, 2, -1])

    def test_all_of_integer(self):
        # TODO
        pass

    def test_all_of_height(self):
        # TODO
        pass

