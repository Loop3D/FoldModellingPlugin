import pandas as pd
from typing import List, Optional, Dict
import numpy as np
import CheckInputData
from fold_modelling_plugin._helper import *


class InputDataProcessor(CheckInputData):

    def __init__(self, data: pd.DataFrame, bounding_box: np.ndarray,
                 knowledge_constraints: Dict = None) -> None:
        """
        Constructs all the necessary attributes for the InputDataProcessor object.

        Parameters
        ----------
        input_data : pd.DataFrame
            The input data to be processed.
        """
        self.data = data
        self.bounding_box = bounding_box
        self.knowledge_constraints = knowledge_constraints

    def process_data(self):
        check_data = CheckInputData(self.data, self.bounding_box, self.knowledge_constraints)
        check_data.check_input_data()  # check the input data is valid

        if ['strike', 'dip'] in self.data.columns:
            strike = self.data['strike'].to_numpy()
            dip = self.data['dip'].to_numpy()
            self.data['gx'], self.data['gy'], self.data['gz'] = strike_dip_to_vector(strike, dip)

        if ['gx', 'gy', 'gz'] in self.data.columns:
            gradient = self.data[['gx', 'gy', 'gz']].to_numpy()
            gradient /= np.linalg.norm(gradient, axis=1)[:, None]  # normalise the gradient vectors
            self.data['gx'], self.data['gy'], self.data['gz'] = gradient[:, 0], gradient[:, 1], gradient[:, 2]

        return self.data
