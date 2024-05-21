from dataclasses import dataclass
from ..input.input_data_checker import CheckInputData
from ..input.data_storage import InputData
from ..datatypes import DataType
from ..utils.utils import strike_dip_to_vectors

import numpy
import pandas
import beartype


@beartype.beartype
@dataclass
class InputDataProcessor:
    data: InputData = None
    processed_data: pandas.DataFrame = None

    def __post_init__(self):
        CheckInputData()(self.data[DataType.DATA])
        if (
            "strike" in self.data[DataType.DATA].columns
            and "dip" in self.data[DataType.DATA].columns
        ):
            strike = self.data[DataType.DATA]["strike"].to_numpy()
            dip = self.data[DataType.DATA]["dip"].to_numpy()
            gradient = strike_dip_to_vectors(strike, dip)

        elif (
            "gx" in self.data[DataType.DATA].columns
            and "gy" in self.data[DataType.DATA].columns
            and "gz" in self.data[DataType.DATA].columns
        ):
            gradient = self.data[DataType.DATA][["gx", "gy", "gz"]].to_numpy()

        gradient = InputDataProcessor.normalise(gradient)

        (
            self.data[DataType.DATA]["gx"],
            self.data[DataType.DATA]["gy"],
            self.data[DataType.DATA]["gz"],
        ) = (
            gradient[:, 0],
            gradient[:, 1],
            gradient[:, 2],
        )
        if isinstance(self.data[DataType.DATA], pandas.DataFrame):
            self.processed_data = self.data[DataType.DATA]
        else:
            raise ValueError("Data must be a pandas DataFrame.")


    @staticmethod
    def normalise(gradient: numpy.ndarray) -> numpy.ndarray:
        """Normalise vectors."""
        return gradient / numpy.linalg.norm(gradient, axis=1)[:, None]
