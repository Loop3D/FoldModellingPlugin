import pandas as pd
import numpy as np
from typing import List, Optional, Dict


class CheckInputData:
    """
    A class used to check the input data for the optimisation.

    ...

    Attributes
    ----------
    folded_foliation_data : pd.DataFrame
        The data related to a folded foliation or bedding
    bounding_box : nd.array
        The bounding box of the model area
    knowledge_constraints : dict
        The knowledge constraints data (default is None)


    Methods
    -------
    check_foliation_data():
        Checks if the foliation data is a pandas dataframe and has the correct columns.
    check_knowledge_constraints():
        Checks if the knowledge constraints is a dictionary and has the correct format.
    check_bounding_box():
        Checks if the bounding box is a numpy array or a list and has the correct format.
    check_input_data():
        Checks all the input data for the optimisation.
    """

    def __init__(self, folded_foliation_data, bounding_box, knowledge_constraints=None):
        """
        Constructs all the necessary attributes for the CheckInputData object.

        """

        self.folded_foliation_data = folded_foliation_data
        self.bounding_box = bounding_box
        self.knowledge_constraints = knowledge_constraints

    def check_foliation_data(self):
        """
        Check the foliation data is a pandas dataframe and has the correct columns: X, Y, Z, feature_name and
        either strike, dip, or gx, gy, gz
        """
        # # check if the foliation data is a pandas dataframe
        if not isinstance(self.folded_foliation_data, pd.DataFrame):
            raise TypeError("Foliation data must be a pandas DataFrame.")
        required_columns = ['X', 'Y', 'Z', 'feature_name']
        if not all(column in self.folded_foliation_data.columns for column in required_columns):
            raise ValueError("Foliation data must have the columns: X, Y, Z, feature_name.")
        if not (all(column in self.folded_foliation_data.columns for column in ['strike', 'dip']) or
                all(column in self.folded_foliation_data.columns for column in ['gx', 'gy', 'gz'])):
            raise ValueError("Foliation data must have either strike, dip or gx, gy, gz columns.")

    # TODO : 1. rewrite check_knowledge_constraints
    #  2. then test it before
    #  3. implementing it in the geological knowledge class
    def check_knowledge_constraints(self):
        """
        Checks if the given nested dictionary is in the correct format.

        Args:
          nested_dict: The nested dictionary to check.

        Returns:
          True if the nested dictionary is in the correct format, False otherwise.
        """

        # Check if the nested dictionary has the correct keys.
        required_keys = [
            'fold_limb_rotation_angle',
            'fold_axis_rotation_angle',
            'fold_axial_surface'
        ]
        for key in required_keys:
            if key not in nested_dict:
                raise ValueError(f'The nested dictionary must have the key "{key}".')

        # Check the format of the 'fold_limb_rotation_angle' dictionary.
        fold_limb_rotation_angle_dict = nested_dict['fold_limb_rotation_angle']
        required_keys_fold_limb_rotation_angle = [
            'tightness',
            'asymmetry',
            'fold_wavelength',
            'axial_traces'
        ]
        for key in required_keys_fold_limb_rotation_angle:
            if key not in fold_limb_rotation_angle_dict:
                raise ValueError(
                    f'The nested dictionary must have the key "{key}" in the '
                    f'"fold_limb_rotation_angle" dictionary.')

        # Check the format of the 'axial_traces' list.
        axial_traces = fold_limb_rotation_angle_dict['axial_traces']
        if not isinstance(axial_traces, list):
            raise ValueError(
                'The "axial_traces" value in the "fold_limb_rotation_angle" dictionary '
                'must be a list.')

        for axial_trace in axial_traces:
            required_keys_axial_trace = ['mu', 'sigma']
            for key in required_keys_axial_trace:
                if key not in axial_trace:
                    raise ValueError(
                        f'The "axial_trace" list must have the key "{key}".')

        # Check the format of the 'fold_axis_rotation_angle' dictionary.
        fold_axis_rotation_angle_dict = nested_dict['fold_axis_rotation_angle']
        required_keys_fold_axis_rotation_angle = ['hinge_angle', 'fold_axis_wavelength']
        for key in required_keys_fold_axis_rotation_angle:
            if key not in fold_axis_rotation_angle_dict:
                raise ValueError(
                    f'The nested dictionary must have the key "{key}" in the '
                    f'"fold_axis_rotation_angle" dictionary.')

        # Check the format of the 'fold_axial_surface' dictionary.
        fold_axial_surface_dict = nested_dict['fold_axial_surface']
        required_keys_fold_axial_surface = ['axial_surface']
        for key in required_keys_fold_axial_surface:
            if key not in fold_axial_surface_dict:
                raise ValueError(
                    f'The nested dictionary must have the key "{key}" in the '
                    f'"fold_axial_surface" dictionary.')

        return True

    # def check_knowledge_constraints(self):
    #     """
    #     Check the knowledge constraints dictionary format
    #        The constraints dictionary should have the following structure:
    #         {
    #             'fold_limb_rotation_angle': {
    #                 'tightness': {'lb':10, 'ub':10, 'mu':10, 'sigma':10, 'w':10},
    #                 'asymmetry': {'lb':10, 'ub':10, 'mu':10, 'sigma':10, 'w':10},
    #                 'fold_wavelength': {'lb':10, 'ub':10, 'mu':10, 'sigma':10, 'w':10},
    #                 'axial_trace_1': {'mu':10, 'sigma':10},
    #                 'axial_traces_2': {'mu':10, 'sigma':10},
    #                 'axial_traces_3': {'mu':10, 'sigma':10},
    #                 'axial_traces_4': {'mu':10, 'sigma':10},
    #             },
    #             'fold_axis_rotation_angle': {
    #                 'hinge_angle': {'lb':10, 'ub':10, 'mu':10, 'sigma':10, 'w':10},
    #                 'fold_axis_wavelength': {'lb':10, 'ub':10, 'mu':10, 'sigma':10, 'w':10},
    #             },
    #             'fold_axial_surface': {
    #                 'axial_surface': {'lb':10, 'ub':10, 'mu':10, 'kappa':10, 'w':10}
    #             }
    #         }
    #             To add more axial traces, use the following format: axial_trace_1, axial_trace_2 etc.
    #     """
    #     if self.knowledge_constraints is not None:
    #         # check if the knowledge constraints is a dictionary
    #         if not isinstance(self.knowledge_constraints, dict):
    #             raise TypeError("Knowledge constraints must be a dictionary.")
    #         # check if the knowledge constraints has one of the keys: tightness, asymmetry,
    #         # fold_wavelength, axial_trace, axial_surface
    #         if not any(key in self.knowledge_constraints for key in ['tightness', 'asymmetry', 'fold_wavelength',
    #                                                                     'axial_trace', 'axial_surface']):
    #             raise ValueError("Knowledge constraints must have one of the keys: tightness, asymmetry, "
    #                              "fold_wavelength, axial_trace, axial_surface.")
    #         # check if the knowledge constraints has the correct format for each key (mu, sigma, w)
    #         if not all(key in self.knowledge_constraints for key in ['mu', 'sigma', 'w']):
    #             raise ValueError("Knowledge constraints must have the following format for each key: "
    #                              "mu, sigma, w.")
    #         else:
    #             for main_key in self.knowledge_constraints:
    #                 if not all(key in self.knowledge_constraints[main_key] for key in ['mu', 'sigma', 'w']):
    #                     raise ValueError("Knowledge constraints must have the following format for each key: "
    #                                      "mu, sigma, w.")

    def check_bounding_box(self):
        """
        check if the bounding_box is an numpy array or a list following this format:
        [[minX, maxX, minY], [maxY, minZ, maxZ]]
        """
        # check if the bounding box is a numpy array or a list
        if not isinstance(self.bounding_box, (np.ndarray, list)):
            raise TypeError("Bounding box must be a numpy array or a list.")
        # check if the bounding box is empty
        if self.bounding_box.size == 0:
            raise ValueError("bounding_box array is empty.")
        # check if the bounding box has the correct format
        if not len(self.bounding_box[0]) == 3 and not len(self.bounding_box[1]) == 3:
            raise ValueError("Bounding box must have the following format: [[minX, maxX, minY], [maxY, minZ, maxZ]]")

    # write a function that checks all the input data
    def check_input_data(self):
        """
        Check the input data for the optimisation
        """
        self.check_bounding_box()
        self.check_foliation_data()
        if self.knowledge_constraints is not None:
            self.check_knowledge_constraints()

        else:
            pass
