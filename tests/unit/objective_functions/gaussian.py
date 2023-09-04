import unittest
import numpy as np
from your_module import (gaussian_log_likelihood, loglikelihood,
                         loglikelihood_axial_surface, loglikelihood_fourier_series)

class TestYourFunctions(unittest.TestCase):

    def test_gaussian_log_likelihood(self):
        self.assertAlmostEqual(
            gaussian_log_likelihood(1.0, 0.0, 1.0), -1.4189385332046727)
        self.assertAlmostEqual(
            gaussian_log_likelihood(2.0, 1.0, 0.5), -2.8378770664093455)
        with self.assertRaises(ValueError):
            gaussian_log_likelihood(1.0, 0.0, 0.0)  # sigma <= 0 should raise ValueError

    def test_loglikelihood(self):
        self.assertAlmostEqual(loglikelihood(1.0, 1.0), -1.4189385332046727)
        self.assertAlmostEqual(loglikelihood(2.0, 2.0), -1.4189385332046727)
        # Add more test cases for different values

    def test_loglikelihood_axial_surface(self):
        self.assertAlmostEqual(loglikelihood_axial_surface(0.0), 3.7590959995303586)
        self.assertAlmostEqual(loglikelihood_axial_surface(np.pi/2), -2.6855758216157585)
        # Add more test cases for different values

    def test_loglikelihood_fourier_series(self):
        rotation_angle = np.array([10.0, 20.0, 30.0])
        fold_frame_coordinate = np.array([1.0, 2.0, 3.0])
        objective_fn = loglikelihood_fourier_series(rotation_angle, fold_frame_coordinate)
        result = objective_fn([0.0, 1.0, 2.0, 3.0])
        self.assertAlmostEqual(result, -3.0726032493525896)
        # Add more test cases for different inputs

if __name__ == '__main__':
    unittest.main()
