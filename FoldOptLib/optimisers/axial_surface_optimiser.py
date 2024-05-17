import gc
from typing import Dict, Any, Union, Callable
import numpy
import pandas
from .fold_optimiser import BaseOptimiser
from ..objective_functions import GeologicalKnowledgeFunctions, ObjectiveFunction
from ..input import CheckInputData, InputData
from ..utils.utils import strike_dip_to_vector, normal_vector_to_strike_and_dip
from ..objective_functions import VonMisesFisher
from ..fold_modelling import FoldModel
from ..datatypes import (

    SolverType, 
    ObjectiveType, 
    OptimisationType, 
    InputGeologicalKnowledge, 
    KnowledgeType, 
    DataType
    )

from LoopStructural import BoundingBox
import beartype


@beartype.beartype
def calculate_intersection_lineation(axial_surface, folded_foliation):
    """
    Calculate the intersection lineation of the axial surface and the folded foliation.

    Parameters:
    axial_surface (np.ndarray): The normal vector of the axial surface.
    folded_foliation (np.ndarray): The normal vector of the folded foliation.

    Returns:
    np.ndarray: The normalised intersection lineation vector.
    """
    # Check if the inputs are numpy arrays
    if not isinstance(axial_surface, numpy.ndarray):
        raise TypeError("Axial surface vector must be a numpy array.")
    if not isinstance(folded_foliation, numpy.ndarray):
        raise TypeError("Folded foliation vector must be a numpy array.")

    # Check if the inputs have the same shape
    if axial_surface.shape != folded_foliation.shape:
        raise ValueError("Axial surface and folded foliation arrays must have the same shape.")

    # Calculate cross product of the axial surface and folded foliation normal vectors
    li = numpy.cross(axial_surface, folded_foliation)

    # Normalise the intersection lineation vector
    li /= numpy.linalg.norm(li, axis=1)[:, None]

    return li


@beartype.beartype
class AxialSurfaceOptimiser(BaseOptimiser):
    """
        Optimiser for Axial Surfaces.

        This class inherits from FoldOptimiser, FoldModel. It is used to optimise the axial surfaces
        based on the provided data, bounding box, geological knowledge.

    """

    def __init__(self, data: InputData,
                 method: str = 'differential_evolution',
                 **kwargs: Dict[str, Any]):

        """
                Initialise the AxialSurfaceOptimiser with data, bounding box, geological knowledge and other parameters.

                Parameters
                ----------
                data : pd.DataFrame
                    The input data for optimisation.
                bounding_box : Union[list, np.ndarray]
                    The bounding box for the optimisation.
                geological_knowledge : Optional[Dict[str, Any]], optional
                    The geological knowledge used for optimisation, by default None.
                **kwargs : Dict[str, Any]
                    Other optional parameters for optimisation. Can include scipy optimisation parameters for
                    differential evolution and trust-constr methods.
                    mode : str, optional, optimisation mode, can be 'restricted' or 'unrestricted',
                    by default 'unrestricted'. only unrestricted mode is supported for now.
                    method : str, optional, optimisation method, can be 'differential_evolution' or 'trust-region',
                    by default 'differential_evolution'.

                """

        super().__init__(method=method)
        self.fold_engine = FoldModel(data, **kwargs)

        self.data = data
        # self.geological_knowledge = geological_knowledge
        self.geological_knowledge = self.setup_geological_knowledge(self.data[DataType.GEOLOGICAL_KNOWLEDGE])
        self.optimisation_type = self.setup_optimisation_type()
        self.gradient_data = self.data[['gx', 'gy', 'gz']].to_numpy()
        self.objective_function = None
        self.guess = None
        self.bounds = None
        self.kwargs = kwargs


    def setup_optimisation_type(self):
        """
        Setup the optimisation type.

        Returns
        -------
        OptimisationType
            The type of optimisation to be performed.
        """
        # Check if geological knowledge is available if not return angle optimisation
        if self.geological_knowledge is None:
            return OptimisationType.ANGLE

        elif self.geological_knowledge is not None:
            for knowledge_type in KnowledgeType:
                if (
                        self.geological_knowledge[knowledge_type] is not None
                        and knowledge_type is KnowledgeType.AXIAL_SURFACE
                ):
                    return OptimisationType.VM_MLE

                elif (
                        self.geological_knowledge[knowledge_type] is not None
                        and knowledge_type != KnowledgeType.AXIAL_SURFACE
                ):
                    return OptimisationType.MLE

    @beartype.beartype
    @staticmethod
    def setup_geological_knowledge(geological_knowledge: InputGeologicalKnowledge = None):

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

    def generate_bounds_and_initial_guess(self):
        """
        Generate the initial guess for the optimisation for differential evolution algorithm.
        The initial guess is generated using the Von Mises Fisher distribution.

        """
        self.bounds = [(0, 360), (0, 90)]

        if self.geological_knowledge[KnowledgeType.AXIAL_SURFACE] is not None:
            # Create a VonMisesFisher distribution with the given parameters
            mu = self.geological_knowledge[KnowledgeType.AXIAL_SURFACE].mu
            kappa = 5
            vmf = VonMisesFisher(mu, kappa)
            # Sample from the distribution
            initial_guess = vmf.draw_samples(size=20, random_state=180)
            initial_guess = normal_vector_to_strike_and_dip(initial_guess)
            return initial_guess

        if self.geological_knowledge[KnowledgeType.AXIAL_SURFACE] is None:
            # use the halton method to initialise the optimisation
            return 'halton'

    @beartype.beartype
    def get_predicted_foliation(self, unit_vector: numpy.ndarray):
        """
        Get the predicted orientation of foliation.

        Parameters
        ----------
        unit_vector: np.ndarray
            The normal unit vector to the axial surface.


        Returns
        -------
        np.ndarray
            The predicted foliation data.
        """

        # Get the predicted foliation based on the unit vector
        predicted_foliation = self.fold_engine.get_predicted_foliation(unit_vector)

        return predicted_foliation

    def setup_optimisation_method(self):
        """
        Sets up the optimisation method.

        Returns
        -------
        Callable
            The objective functions to be minimised.
        """

        # Check if geological knowledge about the axial surface is available
        if self.geological_knowledge is not None:

            # If the optimisation type is MLE, calculate the logpdf using the log normal distribution
            if self.optimisation_type is OptimisationType.MLE:
                self.objective_function = self.build_optimisation_function(
                    ObjectiveFunction[ObjectiveType.LOG_NORMAL],
                    self.get_predicted_foliation,
                    self.geological_knowledge
                )

            # If the optimisation type is VMF_MLE, calculate the logpdf using the Von Mises Fisher distribution
            if self.optimisation_type is OptimisationType.VM_MLE:
                self.objective_function = self.build_optimisation_function(
                    ObjectiveFunction[ObjectiveType.VON_MISES],
                    self.get_predicted_foliation,
                    self.geological_knowledge
                )

        # If no geological knowledge about the axial surface is available
        elif self.geological_knowledge is None:

            # If the optimisation type is MLE, calculate the logpdf using the log normal distribution
            if self.optimisation_type is OptimisationType.MLE:
                self.objective_function = self.build_optimisation_function(
                    ObjectiveFunction[ObjectiveType.LOG_NORMAL],
                    self.get_predicted_foliation
                )

            # If the optimisation type is VMF_MLE, calculate the logpdf using the Von Mises distribution
            elif self.optimisation_type is OptimisationType.VM_MLE:
                self.objective_function = self.build_optimisation_function(
                    ObjectiveFunction[ObjectiveType.VON_MISES],
                    self.get_predicted_foliation
                )

    def build_optimisation_function(
            self,
            objective_function: ObjectiveFunction,
            foliation_function: Callable,
            knowledge_function: GeologicalKnowledgeFunctions = None):

        def optimisation_function(strike_dip):
            # Convert the strike-dip to a unit vector
            unit_vector = strike_dip_to_vector(strike_dip[0], strike_dip[1])
            # Normalize the unit vector
            unit_vector /= numpy.linalg.norm(unit_vector)
            # Get the predicted foliation based on the unit vector
            predicted_foliation = foliation_function(unit_vector)
            # Calculate the angle difference between the predicted and observed foliation
            angle_difference = ObjectiveFunction[ObjectiveType.ANGLE](predicted_foliation, self.gradient_data)
            # clean up memory
            del predicted_foliation, unit_vector
            gc.collect()

            # If the optimisation type is angle, return the angle difference
            if self.optimisation_type == OptimisationType.ANGLE:
                return angle_difference
            else:
                if knowledge_function is None:
                    # Calculate the logpdf of the angle difference
                    logpdf = objective_function(angle_difference)
                    return logpdf

                elif knowledge_function is not None:
                    # Calculate the logpdf of the angle difference
                    logpdf = objective_function(angle_difference) + knowledge_function(unit_vector)
                    return logpdf

        return optimisation_function

    def setup_optimisation(self):

        """
           Sets up the optimisation algorithm, the solver, the objective function, and the initial guess.

        """
        super().setup_optimisation()

        # Generate initial guess
        self.guess = self.generate_bounds_and_initial_guess()

        # Setup optimisation method
        self.setup_optimisation_method()
    
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

        elif self._solver is self.optimiser._solvers[SolverType.CONSTRAINED_TRUST_REGION]:

            return self._solver(self.objective_function, x0=self._guess)

        # TODO: ...add support for restricted optimisation mode...
