import pandas as pd
import beartype



class CheckInputData:
    """
    A class used to check the input data for the optimisation.

    Methods
    -------
    __call__ (folded_foliation_data: pd.DataFrame) -> None
        Checks if the foliation data is a pandas dataframe and has the correct columns.
    """
    # @beartype.beartype
    def __call__(self, folded_foliation_data: pd.DataFrame):
        """
        Check the foliation data is a pandas dataframe and has the correct columns: X, Y, Z, feature_name and
        either strike, dip, or gx, gy, gz
        """
        # # check if the foliation data is a pandas dataframe
        if not isinstance(folded_foliation_data, pd.DataFrame):
            raise TypeError("Foliation data must be a pandas DataFrame.")
        required_columns = ['X', 'Y', 'Z', 'feature_name']
        if not all(column in folded_foliation_data.columns for column in required_columns):
            raise ValueError("Foliation data must have the columns: X, Y, Z, feature_name.")
        if not (all(column in folded_foliation_data.columns for column in ['strike', 'dip']) or
                all(column in folded_foliation_data.columns for column in ['gx', 'gy', 'gz'])):
            raise ValueError("Foliation data must have either strike, dip or gx, gy, gz columns.")