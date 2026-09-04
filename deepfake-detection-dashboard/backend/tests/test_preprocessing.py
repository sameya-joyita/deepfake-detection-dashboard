import unittest

from app.services.preprocessing import uniform_frame_indices


class SamplingTests(unittest.TestCase):
    def test_short_video_uses_every_frame(self):
        self.assertEqual(uniform_frame_indices(3, 20), [0, 1, 2])

    def test_twenty_indices_include_video_endpoints(self):
        indices = uniform_frame_indices(100, 20)
        self.assertEqual(len(indices), 20)
        self.assertEqual(indices[0], 0)
        self.assertEqual(indices[-1], 99)

    def test_invalid_request_is_rejected(self):
        with self.assertRaises(ValueError):
            uniform_frame_indices(100, 0)


if __name__ == "__main__":
    unittest.main()
