import unittest
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, normalize
from FoldModellingPlugin.fold_modelling_plugin.optimisers.fourier_optimiser import FourierSeriesOptimiser
from FoldModellingPlugin.fold_modelling_plugin.helper._helper import *

def scale(data):
    mms = MinMaxScaler()
    mms.fit(data)
    data_transformed = mms.transform(data)

    return data_transformed


def get_wavelength_guesses(guess, size):
    np.random.seed(180)
    mu, sigma = guess, guess / 3
    return np.random.normal(mu, abs(sigma), size)


def objective_wrapper(func1, func2):
    def objective_function(x):
        return func1(x) + func2(x)

    return objective_function

class TestUtilityFunctions(unittest.TestCase):

    def test_scale(self):
        data = np.array([[1, 2], [3, 4], [5, 6]])
        scaled_data = scale(data)
        self.assertTrue(np.all(scaled_data >= 0) and np.all(scaled_data <= 1))

    def test_get_wavelength_guesses(self):
        guesses = get_wavelength_guesses(10, 100)
        self.assertEqual(len(guesses), 100)
        self.assertTrue(np.all(guesses >= 0))

    def test_objective_wrapper(self):
        func1 = lambda x: x ** 2
        func2 = lambda x: x + 2
        wrapped_func = objective_wrapper(func1, func2)
        self.assertEqual(wrapped_func(2), 8)


class TestFourierSeriesOptimiser(unittest.TestCase):

    def setUp(self):
        self.fold_frame_coordinate = np.array([1, 2, 3])
        self.rotation_angle = 45
        self.x = np.array([1, 2, 3])
        self.optimiser = FourierSeriesOptimiser(self.fold_frame_coordinate, self.rotation_angle, self.x)

    def test_prepare_and_setup_knowledge_constraints(self):
        # Test with no geological knowledge
        result = self.optimiser.prepare_and_setup_knowledge_constraints()
        self.assertIsNone(result)

        # Test with geological knowledge (this requires a mock geological knowledge)
        # TODO: Add test with mock geological knowledge

    def test_generate_initial_guess(self):
        # Test with no method and no wl_guess in kwargs
        guess = self.optimiser.generate_initial_guess()
        self.assertIsInstance(guess, np.ndarray)

        # TODO: Add more tests for different kwargs configurations

    def test_setup_optimisation(self):
        obj_func, geo_knowledge, solver, guess = self.optimiser.setup_optimisation()
        self.assertTrue(callable(obj_func))
        self.assertIsNone(geo_knowledge)
        self.assertTrue(callable(solver))
        self.assertIsInstance(guess, np.ndarray)

    def test_optimise(self):
        result = self.optimiser.optimise()
        self.assertIsInstance(result, dict)


if __name__ == '__main__':
    unittest.main()
