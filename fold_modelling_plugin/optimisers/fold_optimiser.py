from typing import Optional, Dict, Any, Union
import pandas as pd
import numpy as np
from ..input.input_data_checker import CheckInputData
from ..helper._helper import *
from ..helper.utils import *
from .base_optimiser import BaseOptimiser
from abc import ABC, abstractmethod
from ..objective_functions.geological_knowledge import GeologicalKnowledgeFunctions


class FoldOptimiser(BaseOptimiser):
    """
    Base class for fold geometry optimisation.
    """

    def __init__(self, **kwargs: Dict[str, Any]):
        """
        Constructs all the necessary attributes for the Fold Optimiser object.

        Parameters
        ----------
            knowledge_constraints : dict, optional
                The knowledge constraints for the optimiser.
            kwargs : dict
                Additional keyword arguments.
        """
        self.kwargs = kwargs

    @abstractmethod
    def prepare_and_setup_knowledge_constraints(self, geological_knowledge) ->\
            Optional[Union[GeologicalKnowledgeFunctions, None]]:
        """
        Prepare the knowledge constraints data
        """
        # Check if knowledge constraints exist
        if geological_knowledge is not None:
            # TODO: Add a check if the knowledge constraints are in the correct format
            # Check if mode is restricted
            if 'mode' in self.kwargs and self.kwargs['mode'] == 'restricted':
                geological_knowledge = GeologicalKnowledgeFunctions(geological_knowledge)
                ready_constraints = geological_knowledge.setup_objective_functions_for_restricted_mode(self)

                return ready_constraints
            else:
                geological_knowledge = GeologicalKnowledgeFunctions(geological_knowledge)

                return geological_knowledge

        # If knowledge constraints do not exist, return None

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
    def setup_optimisation(self, geological_knowledge=None) -> tuple:
        """
        Setup optimisation.

        Returns
        -------
        tuple
            Returns a tuple containing the geological knowledge objective functions, a solver, and the initial guess.
        """

        # Check if method is specified in kwargs and assign the appropriate solver
        if 'method' in self.kwargs and self.kwargs['method'] == 'differential_evolution':
            solver = self.optimise_with_differential_evolution
        else:
            solver = self.optimise_with_trust_region

        # Prepare and setup knowledge constraints
        geological_knowledge = self.prepare_and_setup_knowledge_constraints(geological_knowledge)

        return geological_knowledge, solver

    @abstractmethod
    def optimise(self, *args, **kwargs):
        """
        Run the optimisation
        """

        pass
