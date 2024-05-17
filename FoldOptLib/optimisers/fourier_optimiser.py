from typing import Union, Any, Dict
from ..datatypes import ObjectiveType, InputGeologicalKnowledge, SolverType
from .fold_optimiser import BaseOptimiser
from ..objective_functions import ObjectiveFunction, GeologicalKnowledgeFunctions
from ..utils.utils import (
    calculate_semivariogram,
    get_wavelength_guesses,
)
import numpy


class FourierSeriesOptimiser(BaseOptimiser):
    """
    A class used to represent a Fourier Series Optimiser.

    ...

    Attributes
    ----------
    fold_frame_coordinate : float
        The fold frame coordinate for the optimiser.
    rotation_angle : float
        The rotation angle for the optimiser.
    knowledge_constraints : dict, optional
        The knowledge constraints for the optimiser.
    x : float
        The interpolated fold frame coordinate z or y: np.linspace(z.min(), z.max(), 100).
        It's used to calculate the optimised Fourier series everywhere in the model space.
    kwargs : dict
        Additional keyword arguments.

    Methods
    -------
    TODO: Add methods here.
    """

    def __init__(
        self,
        fold_frame_coordinate: Union[list, numpy.ndarray],
        rotation_angle: Union[list, numpy.ndarray],
        x: Union[list, numpy.ndarray],
        geological_knowledge: GeologicalKnowledgeFunctions = None,
        method="differential_evolution",
        **kwargs: Dict[str, Any],
    ):
        """
        Constructs all the necessary attributes for the Fourier Series Optimiser object.

        Parameters
        ----------
            fold_frame_coordinate : float
                The fold frame coordinate for the optimiser.
            rotation_angle : float
                The rotation angle for the optimiser.
            geological_knowledge : dict, optional
                The knowledge constraints for the optimiser.
            x : np.ndarray
                The interpolated fold frame coordinate z or y: np.linspace(z.min(), z.max(), 100).
                It's used to calculate the optimised Fourier series everywhere in the model space.
            **kwargs : dict
                Additional keyword arguments.
        """
        BaseOptimiser.__init__(self, method=method, **kwargs)
        self.objective_value = 0
        self.fold_frame_coordinate = fold_frame_coordinate
        self.rotation_angle = numpy.tan(numpy.deg2rad(rotation_angle))
        self.method = method
        self.geological_knowledge = self.setup_geological_knowledge(
            geological_knowledge
        )
        # TODO: check how to initialise self.x = x in self.geological_knowledge
        self.x = x
        self.solver = None
        self.objective_function = None
        self.guess = None
        self.bounds = None
        self.kwargs = kwargs

    def generate_bounds_and_initial_guess(self) -> Union[numpy.ndarray, Any]:
        """
        Generate an initial guess for the optimisation.
        It generates a guess of the wavelength for the Fourier series. The format of the guess depends
        on the method of optimisation - differential evolution or trust region. If it's differential evolution,
        it will generate the bounds for the optimisation. If it's trust region, it will generate the initial guess of
        the wavelength.

        Returns
        -------
        np.ndarray or Any
            Returns the initial guess or bounds for the optimisation.
        """

        # Check if method is specified in kwargs
        if self.method == "differential_evolution":
            if "wl_guess" in self.kwargs:
                wl = get_wavelength_guesses(self.kwargs["wl_guess"], 1000)
                # bounds = np.array([(-1, 1), (-1, 1), (-1, 1), (wl[wl > 0].min() / 2, wl.max())], dtype=object)
                self.bounds = numpy.array(
                    [(-1, 1), (-1, 1), (-1, 1), (wl[wl > 0].min() / 2, wl.max())],
                    dtype=object,
                )

            else:
                # Calculate semivariogram and get the wavelength guess
                guess, lags, variogram = calculate_semivariogram(
                    self.fold_frame_coordinate, self.rotation_angle
                )
                wl = get_wavelength_guesses(guess[3], 1000)
                self.bounds = numpy.array(
                    [(-1, 1), (-1, 1), (-1, 1), (wl[wl > 0].min() / 2, wl.max() / 3)],
                    dtype=object,
                )

        # Check if wl_guess is specified in kwargs
        if "wl_guess" in self.kwargs:
            self.guess = numpy.array([0, 1, 1, self.kwargs["wl_guess"]], dtype=object)

        else:
            # Calculate semivariogram and get the wavelength guess
            self.guess, lags, variogram = calculate_semivariogram(
                self.fold_frame_coordinate, self.rotation_angle
            )

    @staticmethod
    def setup_geological_knowledge(
        geological_knowledge: InputGeologicalKnowledge = None,
    ):
        """
        Setup the geological knowledge.

        Returns
        -------
        GeologicalKnowledgeFunctions
            The geological knowledge functions.
        """
        if geological_knowledge is not None:
            return GeologicalKnowledgeFunctions(geological_knowledge)
        else:
            return None

    def build_optimisation_function(
        self,
        objective_function: ObjectiveFunction,
        knowledge_function: GeologicalKnowledgeFunctions = None,
    ):
        def optimisation_function(theta):
            return objective_function(theta) + knowledge_function(theta)

        return optimisation_function

    def setup_optimisation_method(self):
        """
        Sets up the optimisation method.

        Returns
        -------
        Callable
            The objective functions to be minimised.
        """
        # for the moment fourier series optimisation is done only
        # using MLE optimisation. Least squares will be added later
        if self.geological_knowledge is not None:
            self.objective_function = self.build_optimisation_function(
                ObjectiveFunction[ObjectiveType.FOURIER](
                    self.rotation_angle, self.fold_frame_coordinate
                ),
                self.geological_knowledge,
            )

        elif self.geological_knowledge is None:
            self.objective_function = ObjectiveFunction[ObjectiveType.FOURIER](
                self.rotation_angle, self.fold_frame_coordinate
            )

    def setup_optimisation(self):
        """
        Setup Fourier series optimisation.

        Returns
        -------
        tuple
            Returns a tuple containing the objective function, geological knowledge, solver, and initial guess.
        """

        # # Check if method is specified in kwargs and assign the appropriate solver
        # if 'method' in self.kwargs and self.kwargs['method'] == 'differential_evolution':
        #     solver = self.optimise_with_differential_evolution
        # else:
        #     solver = self.optimise_with_trust_region

        # Setup objective function
        self.setup_optimisation_method()

        # Prepare and setup knowledge constraints
        super().setup_optimisation()

        # Generate initial guess
        self.generate_bounds_and_initial_guess()

    def optimise(self):
        """
        Runs the optimisation.

        Returns
        -------
        opt : Dict
            The result of the optimisation.

        Notes
        -----
        This function runs the optimisation by setting up the optimisation problem,
        checking if geological knowledge exists, and running the solver.
        """

        self.setup_optimisation()

        if self._solver is self.optimiser._solvers[SolverType.DIFFERENTIAL_EVOLUTION]:
            return self._solver(self.objective_function, self._bounds, init=self._guess)

        elif (
            self._solver is self.optimiser._solvers[SolverType.CONSTRAINED_TRUST_REGION]
        ):
            return self._solver(self.objective_function, x0=self._guess)

        # TODO: ...add support for restricted optimisation mode...
