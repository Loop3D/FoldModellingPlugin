from abc import ABC, abstractmethod
import numpy as np


class BaseEngine(ABC):

    @abstractmethod
    def initialise_model(self):
        """
        Setup the bounding box of the fold frame
        """
        pass

    @abstractmethod
    def build_fold_frame(self, axial_normal: np.ndarray) -> None:
        """
        Build the fold frame
        """
        pass
