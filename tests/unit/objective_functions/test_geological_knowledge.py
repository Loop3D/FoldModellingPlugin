import unittest
import numpy as np

# Import the class to be tested
from FoldModellingPlugin.fold_modelling_plugin.objective_functions.geological_knowledge import \
    GeologicalKnowledgeFunctions


class TestGeologicalKnowledgeFunctions(unittest.TestCase):

    def setUp(self):
        # Create an instance of GeologicalKnowledgeFunctions with dummy constraints and data
        self.constraints = {
            'fold_limb_rotation_angle': {
                'tightness': {'lb': 10, 'ub': 10, 'mu': 10, 'sigma': 10, 'w': 10},
                'asymmetry': {'lb': 10, 'ub': 10, 'mu': 10, 'sigma': 10, 'w': 10},
                'fold_wavelength': {'lb': 10, 'ub': 10, 'mu': 10, 'sigma': 10, 'w': 10},
                'axial_trace_1': {'mu': 10, 'sigma': 10},
                'axial_traces_2': {'mu': 10, 'sigma': 10},
                'axial_traces_3': {'mu': 10, 'sigma': 10},
                'axial_traces_4': {'mu': 10, 'sigma': 10},
            },
            'fold_axis_rotation_angle': {
                'hinge_angle': {'lb': 10, 'ub': 10, 'mu': 10, 'sigma': 10, 'w': 10},
                'fold_axis_wavelength': {'lb': 10, 'ub': 10, 'mu': 10, 'sigma': 10, 'w': 10},
            },
            'fold_axial_surface': {
                'axial_surface': {'lb': 10, 'ub': 10, 'mu': 10, 'kappa': 10, 'w': 10}
            }
        }
        self.x = np.linspace(0, 10, 100)
        self.geo_knowledge = GeologicalKnowledgeFunctions(self.constraints, x=self.x)

    def test_axial_surface_objective_function(self):
        # Test if axial_surface_objective_function raises KeyError when constraints are missing
        with self.assertRaises(KeyError):
            self.geo_knowledge.axial_surface_objective_function(np.array([0, 1, 0]))

        # Test if axial_surface_objective_function returns a float
        result = self.geo_knowledge.axial_surface_objective_function(np.array([0.5, 0.5, 0.5]))
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
