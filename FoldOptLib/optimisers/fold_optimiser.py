from typing import Dict, Any
from abc import ABC, abstractmethod
from ..objective_functions.geological_knowledge import GeologicalKnowledgeFunctions
from ..solvers import Solver
from ..datatypes import SolverType, InputGeologicalKnowledge


class BaseOptimiser(ABC):
    """
    Base class for configuration of fold geometry optimisers.
    """

    def __init__(self, method: str = 'differential_evolution', **kwargs: Dict[str, Any]):
        """
        Constructs all the necessary attributes for the Fold Optimiser object.

        Parameters
        ----------
            kwargs : dict
                Additional keyword arguments.
        """
        self.method = method
        self._solvers = Solver()
        self.solver = None
        self.kwargs = kwargs

    @staticmethod
    @abstractmethod
    def setup_geological_knowledge(geological_knowledge: InputGeologicalKnowledge = None):

        """
        Setup the geological knowledge.

        Returns
        -------
        GeologicalKnowledgeFunctions
            The geological knowledge functions.
        """
        pass

    @abstractmethod
    def generate_bounds_and_initial_guess(self):
        """
        Generate an initial guess for the optimisation
        It generates a guess depending on the type of optimisation, if it's fourier series
        it will generate a guess of the wavelength, if it's axial surface it will generate a guess
        using the methods of the Differential Evolution algorithm (Stern and Price, 1997) or uses the
        Von Mises Fisher distribution (Fisher, 1953).
        """

        pass

    @abstractmethod
    def build_optimisation_function(self, *args):
        """
        Build the optimisation function.
        Returns
        -------

        """
        pass

    @abstractmethod
    def setup_optimisation_method(self):
        """
        Setup optimisation method.
        """

        pass

    @abstractmethod
    def setup_optimisation(self):
        """
        Setup optimisation.

        Returns
        -------
        tuple
            Returns a tuple containing the geological knowledge objective functions, a solver, and the initial guess.
        """

        # Check if method is specified in kwargs and assign the appropriate solver
        if self.method == 'differential_evolution':
            self.solver = self._solvers[SolverType.DIFFERENTIAL_EVOLUTION]

        else:
            self.solver = self._solvers[SolverType.CONSTRAINED_TRUST_REGION]

    @abstractmethod
    def optimise(self, *args, **kwargs):
        """
        Run the optimisation
        """

        pass
