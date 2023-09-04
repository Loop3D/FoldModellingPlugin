import unittest
import numpy as np
from fold_modelling_plugin.objective_functions.axial_plane.py import is_axial_plane_compatible


class TestIsAxialPlaneCompatible(unittest.TestCase):
    def test_raises_value_error_when_v1_and_v2_are_not_numpy_arrays(self):
        with self.assertRaises(ValueError):
            is_axial_plane_compatible([1, 2, 3], "hello")

    def test_raises_value_error_when_v1_and_v2_have_different_shapes(self):
        v1 = np.array([1, 2, 3])
        v2 = np.array([4, 5, 6, 7])
        with self.assertRaises(ValueError):
            is_axial_plane_compatible(v1, v2)

    def test_returns_zero_when_v1_and_v2_are_the_same(self):
        v1 = np.array([1, 0, 0])
        v2 = v1
        objective_func = is_axial_plane_compatible(v1, v2)
        self.assertEqual(objective_func, 0)

    def test_returns_the_angle_difference_between_v1_and_v2(self):
        v1 = np.array([1, 0, 0])
        v2 = np.array([0, 1, 0])
        objective_func = is_axial_plane_compatible(v1, v2)
        self.assertEqual(objective_func, np.pi / 2)


if __name__ == "__main__":
    unittest.main()

