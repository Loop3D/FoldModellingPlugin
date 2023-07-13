from typing import Optional, Dict, Any
import pandas as pd
import numpy as np
from fold_modelling_plugin.input.input_data_checker import CheckInputData
from base_optimiser import BaseOptimiser
from abc import ABC, abstractmethod


class FoldOptimiser(ABC, BaseOptimiser):
    """
    Base class for fold geometry optimisation.
    """

    @abstractmethod
    def prepare_and_setup_knowledge_constraints(self):
        """
        Prepare the knowledge constraints data
        """
        pass

    @abstractmethod
    def generate_initial_guess(self):
        """
        Generate an initial guess for the optimisation
        It generates a guess depending on the type of optimisation, if it's fourier series
        it will generate a guess of the wavelength, if it's axial surface it will generate a guess
        using the methods of the Differential Evolution algorithm (Storn and Price, 1997) or uses the
        Von Mises Fisher distribution (Fisher, 1953).
        """

        pass

    @abstractmethod
    def setup_optimisation(self):
        """
        Setup the optimisation problem
        """
        pass

    @abstractmethod
    def optimise(self, *args, **kwargs):
        """
        Run the optimisation
        """

        pass
