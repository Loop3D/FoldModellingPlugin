import unittest
import pandas as pd
import numpy as np
from FoldModellingPlugin.fold_modelling_plugin.helper.utils import *
from FoldModellingPlugin.fold_modelling_plugin.input.input_data_processor import InputDataProcessor


class TestInputDataProcessor(unittest.TestCase):

    def test_init(self):
        data = pd.DataFrame({
            'X': [1, 2, 3],
            'Y': [4, 5, 6],
            'Z': [7, 8, 9],
            'feature_name': ['s0', 's0', 's0'],
            'gx': [0.1, 0.2, 0.3],
            'gy': [0.4, 0.5, 0.6],
            'gz': [0.7, 0.8, 0.9]
        })
        bounding_box = np.array([[0, 0, 0], [10, 10, 10]])
        knowledge_constraints = None
        input_data_processor = InputDataProcessor(data, bounding_box, knowledge_constraints)
        self.assertEqual(input_data_processor.data.all, data.all)
        self.assertEqual(np.all(input_data_processor.bounding_box), np.all(bounding_box))
        self.assertEqual(input_data_processor.knowledge_constraints, knowledge_constraints)
        c = ['X', 'Y', 'Z', 'feature_name', 'strike', 'dip', 'gx', 'gy', 'gz']

    def test_process_data_with_strike_dip(self):
        data = data = pd.DataFrame({
            'X': [1, 2, 3],
            'Y': [4, 5, 6],
            'Z': [7, 8, 9],
            'feature_name': ['s0', 's0', 's0'],
            'strike': [0.1, 0.2, 0.3],
            'dip': [0.4, 0.5, 0.6],
        })
        bounding_box = np.array([[0, 0, 0], [10, 10, 10]])
        knowledge_constraints = None
        input_data_processor = InputDataProcessor(data, bounding_box, knowledge_constraints)
        input_data_processor.process_data()
        expected_data = pd.DataFrame({'strike': [90], 'dip': [45], 'gx': [1], 'gy': [0], 'gz': [0.707]})
        c = ['X', 'Y', 'Z', 'feature_name', 'strike', 'dip', 'gx', 'gy', 'gz']
        # self.assertEqual(input_data_processor.data, expected_data)
        self.assertEqual(input_data_processor.data.columns.tolist(), c)


    def test_process_data_without_strike_dip(self):
        data = pd.DataFrame()
        bounding_box = np.array([])
        knowledge_constraints = None
        input_data_processor = InputDataProcessor(data, bounding_box, knowledge_constraints)
        input_data_processor.process_data()
        expected_data = data
        self.assertEqual(input_data_processor.data, expected_data)

    def test_process_data_with_gradient(self):
        data = pd.DataFrame({'gx': [1], 'gy': [0], 'gz': [0.707]})
        bounding_box = np.array([])
        knowledge_constraints = None
        input_data_processor = InputDataProcessor(data, bounding_box, knowledge_constraints)
        input_data_processor.process_data()
        expected_data = pd.DataFrame({'gx': [1], 'gy': [0], 'gz': [0.707]})
        self.assertEqual(input_data_processor.data, expected_data)


if __name__ == "__main__":
    unittest.main()
