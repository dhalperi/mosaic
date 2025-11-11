"""Tests for mosaic.deduper module."""

import unittest

from mosaic.deduper import idx_to_ij


class TestDeduperFunctions(unittest.TestCase):
    """Test the deduper module functions."""

    def test_idx_to_ij_basic(self):
        """Test idx_to_ij with basic cases."""
        # For N=3, the pairs are: (0,1), (0,2), (1,2)
        # idx 0 -> (0,1)
        i, j = idx_to_ij(0, 3)
        self.assertEqual(i, 0)
        self.assertEqual(j, 1)

        # idx 1 -> (0,2)
        i, j = idx_to_ij(1, 3)
        self.assertEqual(i, 0)
        self.assertEqual(j, 2)

        # idx 2 -> (1,2)
        i, j = idx_to_ij(2, 3)
        self.assertEqual(i, 1)
        self.assertEqual(j, 2)

    def test_idx_to_ij_larger(self):
        """Test idx_to_ij with a larger N."""
        # For N=5, pairs: (0,1), (0,2), (0,3), (0,4), (1,2), (1,3), (1,4), (2,3), (2,4), (3,4)
        # idx 4 -> (1,2) - 5th pair
        i, j = idx_to_ij(4, 5)
        self.assertEqual(i, 1)
        self.assertEqual(j, 2)

    def test_idx_to_ij_ensures_i_less_than_j(self):
        """Test that idx_to_ij always returns i < j."""
        for N in range(2, 10):
            num_pairs = N * (N - 1) // 2
            for idx in range(num_pairs):
                i, j = idx_to_ij(idx, N)
                self.assertLess(i, j, f"For N={N}, idx={idx}: i={i} should be < j={j}")
                self.assertGreaterEqual(i, 0)
                self.assertLess(j, N)


if __name__ == "__main__":
    unittest.main()
