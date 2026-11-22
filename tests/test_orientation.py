import sys
import unittest
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from orientation_metrics import angles


class OrientationTests(unittest.TestCase):
    def test_known_roll_and_pitch(self):
        half = .2 / 2
        roll, pitch = angles(np.array([[1, 0, 0, 0], [np.cos(half), np.sin(half), 0, 0],
                                       [np.cos(half), 0, np.sin(half), 0]]))
        np.testing.assert_allclose(roll, [0, .2, 0], atol=1e-12)
        np.testing.assert_allclose(pitch, [0, 0, .2], atol=1e-12)
