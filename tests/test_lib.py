"""Tests for mosaic.lib module."""

import unittest

from mosaic.lib import Thumb


class TestThumb(unittest.TestCase):
    """Test the Thumb named tuple."""

    def test_thumb_creation(self):
        """Test creating a Thumb instance."""
        import numpy as np

        bytes_data = np.array([1, 2, 3, 4], dtype=np.uint8)
        thumb = Thumb(uid="test123", bytes=bytes_data, flipped=False)

        self.assertEqual(thumb.uid, "test123")
        self.assertEqual(len(thumb.bytes), 4)
        self.assertFalse(thumb.flipped)

    def test_thumb_flipped(self):
        """Test creating a flipped Thumb instance."""
        import numpy as np

        bytes_data = np.array([1, 2, 3, 4], dtype=np.uint8)
        thumb = Thumb(uid="test456", bytes=bytes_data, flipped=True)

        self.assertEqual(thumb.uid, "test456")
        self.assertTrue(thumb.flipped)


if __name__ == "__main__":
    unittest.main()
