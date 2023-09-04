import unittest
import numpy as np

# Import the class to be tested
from your_module import GeologicalKnowledgeFunctions

class TestGeologicalKnowledgeFunctions(unittest.TestCase):

    def setUp(self):
        # Create an instance of GeologicalKnowledgeFunctions with dummy constraints and data
        self.constraints = {
            'asymmetry': {'lb': 0, 'ub': 10, 'mu': 5, 'sigma': 1, 'w': 1},
            'tightness': {'lb': 0, 'ub': 10, 'mu': 5, 'sigma': 1, 'w': 1},
            'fold_wavelength': {'lb': 0, 'ub': 10, 'mu': 5, 'sigma': 1, 'w': 1},
        }
        self.data = np.array([0, 1, 2, 3, 4])
        self.geo_knowledge = GeologicalKnowledgeFunctions(self.constraints, x=self.data)

    def test_axial_surface_objective_function(self):
        # Test if axial_surface_objective_function raises KeyError when constraints are missing
        with self.assertRaises(KeyError):
            self.geo_knowledge.axial_surface_objective_function(np.array([0, 1]))

        # Test if axial_surface_objective_function returns a float
        result = self.geo_knowledge.axial_surface_objective_function(np.array([0.5, 0.5]))
        self.assertIsInstance(result, float)

    def test_axial_trace_objective_function(self):
        # Test if axial_trace_objective_function returns a float
        theta = np.array([1, 2, 3, 4])
        result = self.geo_knowledge.axial_trace_objective_function(theta)
        self.assertIsInstance(result, float)

    def test_wavelength_objective_function(self):
        # Test if wavelength_objective_function returns a float
        theta = np.array([1, 2, 3, 4])
        result = self.geo_knowledge.wavelength_objective_function(theta)
        self.assertIsInstance(result, float)

    def test_setup_objective_functions(self):
        # Test if setup_objective_functions creates the objective_functions_map
        self.geo_knowledge.setup_objective_functions()
        self.assertIsNotNone(self.geo_knowledge.objective_functions_map)

    def tearDown(self):
        # Clean up resources if needed
        pass

if __name__ == '__main__':
    unittest.main()


