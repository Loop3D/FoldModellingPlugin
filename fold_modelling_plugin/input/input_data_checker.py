import pandas as pd
import numpy as np


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
        # if not isinstance(self.folded_foliation_data, pd.DataFrame):
        #     raise TypeError("Folded foliation data must be a pandas dataframe.")
        # # check if the foliation data has  X, Y, Z, feature_name columns
        # if not all(col in self.folded_foliation_data.columns for col in ['X', 'Y', 'Z', 'feature_name']):
        #     raise ValueError("Folded foliation data must have the columns: X, Y, Z, feature_name.")
        # # check if the foliation data has either strike, dip or gx, gy, gz columns
        # if not all(col in self.folded_foliation_data.columns for col in ['strike', 'dip']) or \
        #         not all(col in self.folded_foliation_data.columns for col in ['gx', 'gy', 'gz']):
        #     raise ValueError("Folded foliation data must have either strike, dip or gx, gy, gz columns.")
        if not isinstance(self.folded_foliation_data, pd.DataFrame):
            raise TypeError("Foliation data must be a pandas DataFrame.")
        required_columns = ['X', 'Y', 'Z', 'feature_name']
        if not all(column in self.folded_foliation_data.columns for column in required_columns):
            raise ValueError("Foliation data must have the columns: X, Y, Z, feature_name.")
        if not (all(column in self.folded_foliation_data.columns for column in ['strike', 'dip']) or
                all(column in self.folded_foliation_data.columns for column in ['gx', 'gy', 'gz'])):
            raise ValueError("Foliation data must have either strike, dip or gx, gy, gz columns.")

        return True

    def check_knowledge_constraints(self):
        """
        Check the knowledge constraints dictionary format
           The constraints dictionary should have the following structure:
            dict(
                {
                    'tightness': { 'mu':, 'sigma':, 'w':},
                    'asymmetry': { 'mu':, 'sigma':, 'w':},
                    'fold_wavelength': { 'mu':, 'sigma':, 'w':},
                    'axial_trace': {'mu':, 'sigma':, 'w':},
                    'axial_surface':{'mu':, 'sigma':, 'w':},
                })
                To add more axial traces, use the following format: axial_trace_1, axial_trace_2 etc.
        """
        if self.knowledge_constraints is not None:
            # check if the knowledge constraints is a dictionary
            if not isinstance(self.knowledge_constraints, dict):
                raise TypeError("Knowledge constraints must be a dictionary.")
            # check if the knowledge constraints has one of the keys: tightness, asymmetry,
            # fold_wavelength, axial_trace, axial_surface
            if not any(key in self.knowledge_constraints for key in ['tightness', 'asymmetry', 'fold_wavelength',
                                                                     'axial_trace', 'axial_surface']):
                raise ValueError("Knowledge constraints must have one of the keys: tightness, asymmetry, "
                                 "fold_wavelength, axial_trace, axial_surface.")
            # check if the knowledge constraints has the correct format for each key (mu, sigma, w)
            if not all(key in self.knowledge_constraints for key in ['mu', 'sigma', 'w']):
                raise ValueError("Knowledge constraints must have the following format for each key: "
                                 "mu, sigma, w.")
            else:
                for main_key in self.knowledge_constraints:
                    if not all(key in self.knowledge_constraints[main_key] for key in ['mu', 'sigma', 'w']):
                        raise ValueError("Knowledge constraints must have the following format for each key: "
                                         "mu, sigma, w.")

    def check_bounding_box(self):
        """
        check if the bounding_box is an numpy array or a list following this format:
        [[minX, maxX, minY], [maxY, minZ, maxZ]]
        """
        # check if the bounding box is a numpy array or a list
        if not isinstance(self.bounding_box, (np.ndarray, list)):
            raise TypeError("Bounding box must be a numpy array or a list.")
        # check if the bounding box has the correct format
        if not len(self.bounding_box[0]) == 3 or not len(self.bounding_box[1]) == 3:
            raise ValueError("Bounding box must have the following format: [[minX, maxX, minY], [maxY, minZ, maxZ]]")

    # write a function that checks all the input data for the optimisation
    def check_input_data(self):
        """
        Check the input data for the optimisation
        """
        self.check_foliation_data()
        self.check_knowledge_constraints()
        self.check_bounding_box()

# write test function for the class CheckInputData in the file test_input_data_checker.py in the folder tests
