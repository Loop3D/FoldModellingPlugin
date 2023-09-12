from abc import ABC
from typing import Dict, Any, Optional, Union, Tuple

# from scipy.optimize import minimize
# import knowledge_constraints
# importlib.reload(knowledge_constraints)
# from modified_loopstructural.modified_foldframe import FoldFrame
# from modified_loopstructural.extra_utils import *
# from knowledge_constraints._helper import *
# from knowledge_constraints.knowledge_constraints import GeologicalKnowledgeConstraints
# from knowledge_constraints.splot_processor import SPlotProcessor
# from knowledge_constraints.fourier_optimiser import FourierSeriesOptimiser
# from LoopStructural import GeologicalModel
# from LoopStructural.modelling.features.fold import FoldEvent
# from LoopStructural.modelling.features.fold import FoldRotationAngle, SVariogram
# from LoopStructural.modelling.features.fold import fourier_series
# from LoopStructural.helper.helper import *
# from geological_sampler.sampling_methods import *
# from uncertainty_quantification.fold_uncertainty import *
import numpy as np
import pandas as pd
# import mplstereonet
# from sklearn.preprocessing import StandardScaler
# from scipy.optimize import minimize, differential_evolution
# from scipy.stats import vonmises_fisher, vonmises
from .fold_optimiser import FoldOptimiser
from ..objective_functions import GeologicalKnowledgeFunctions
from ..input import CheckInputData
from ..helper.utils import strike_dip_to_vector, normal_vector_to_strike_and_dip
from ..objective_functions import VonMisesFisher
from ..objective_functions import is_axial_plane_compatible
# from ..fold_modelling import FoldModel
from ..objective_functions import loglikelihood_axial_surface


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
    if not isinstance(axial_surface, np.ndarray):
        raise TypeError("Axial surface vector must be a numpy array.")
    if not isinstance(folded_foliation, np.ndarray):
        raise TypeError("Folded foliation vector must be a numpy array.")

    # Check if the inputs have the same shape
    if axial_surface.shape != folded_foliation.shape:
        raise ValueError("Axial surface and folded foliation arrays must have the same shape.")

    # Calculate cross product of the axial surface and folded foliation normal vectors
    intesection_lineation = np.cross(axial_surface, folded_foliation)

    # Normalise the intersection lineation vector
    intesection_lineation /= np.linalg.norm(intesection_lineation, axis=1)[:, None]

    return intesection_lineation


# def logp(value: TensorVariable, mu: TensorVariable) -> TensorVariable:
#     return -(value - mu)**2


class AxialSurfaceOptimiser(FoldOptimiser):
    """
        Optimiser for Axial Surfaces.

        This class inherits from FoldOptimiser, FoldModel. It is used to optimise the axial surfaces
        based on the provided data, bounding box, geological knowledge.

    """

    def __init__(self, data: pd.DataFrame,
                 bounding_box: Union[list, np.ndarray],
                 geological_knowledge: Optional[Dict[str, Any]] = None,
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
                    by default 'unrestricted'.
                    method : str, optional, optimisation method, can be 'differential_evolution' or 'trust-region',
                    by default 'differential_evolution'.

                """

        # Check the input data
        check_input = CheckInputData(data, bounding_box,
                                     knowledge_constraints=geological_knowledge, **kwargs)
        check_input.check_input_data()

        FoldOptimiser.__init__(self, **kwargs)
        # FoldModel.__init__(data, bounding_box, geological_knowledge=geological_knowledge, **kwargs)

        self.data = data
        self.bounding_box = bounding_box
        self.geological_knowledge = geological_knowledge
        self.kwargs = kwargs
        self.gradient_data = self.data[['gx', 'gy', 'gz']].to_numpy()
        self.geo_objective = None
        self.objective_function = None
        self.guess = None
        self.solver = None

    def loglikelihood(self, x, predicted_foliation: np.ndarray,
                      geological_knowledge: GeologicalKnowledgeFunctions) -> float:
        """
         Calculate the maximum likelihood estimate of the axial surface and the geological knowledge.

         Parameters
         ----------
         x : np.ndarray
             The axial surface normal vector to be optimised.
         predicted_foliation : np.ndarray
             The predicted foliation data.
         geological_knowledge : GeologicalKnowledgeFunctions
             The geological knowledge functions.

         Returns
         -------
         float
             The calculated loglikelihood of the axial surface. Returns None if input is not valid.
         """

        # Calculate the angle difference between the predicted and observed foliation
        angle_difference = is_axial_plane_compatible(predicted_foliation, self.gradient_data)
        # Calculate the loglikelihood of the axial surface
        loglikelihood = loglikelihood_axial_surface(angle_difference) + geological_knowledge(x)

        return loglikelihood

    def mle_optimisation(self, strike_dip: Tuple[float, float]):
        """
        Performs Maximum Likelihood Estimation (MLE) optimisation.

        Parameters
        ----------
        strike_dip : tuple
            A tuple containing strike and dip values of the estimated axial surface.

        Returns
        -------
        logpdf : float
            The log-likelihood of the MLE.

        Notes
        -----
        This function performs MLE optimisation and used when geological knowledge constraints are provided.
        The function returns the log-likelihood of the MLE that is optimised
        """

        axial_normal = strike_dip_to_vector(*strike_dip)
        axial_normal /= np.linalg.norm(axial_normal)

        predicted_foliation = self.get_predicted_foliation(strike_dip)
        logpdf = -self.loglikelihood(axial_normal, predicted_foliation, self.geo_objective)
        return logpdf

    def angle_optimisation(self, strike_dip: Tuple[float, float]):
        """
            Minimises the angle between the observed and predicted folded foliation.

            Parameters
            ----------
            strike_dip : tuple
                A tuple containing strike and dip values of the estimated axial surface.

            Returns
            -------
            angle_difference : float
                The difference between the predicted and actual angle.

            Notes
            -----
            This function optimises the axial surface by minimising the angle between the predicted and observed folded
            foliation. This function is used when there are no geological knowledge constraints provided.
        """

        axial_normal = strike_dip_to_vector(*strike_dip)
        axial_normal /= np.linalg.norm(axial_normal)
        predicted_foliation = self.get_predicted_foliation(axial_normal)
        angle_difference = is_axial_plane_compatible(predicted_foliation, self.gradient_data)
        return angle_difference

    def generate_initial_guess(self):
        """
        Generate the initial guess for the optimisation for differential evolution algorithm.
        The initial guess is generated using the Von Mises Fisher distribution.

        """
        if 'axial_surface_guess' in self.kwargs:
            guess = self.kwargs['axial_surface_guess']
            if len(guess) == 2:
                # Create a VonMisesFisher distribution with the given parameters
                mu = strike_dip_to_vector(*guess)
                kappa = 5
                vmf = VonMisesFisher(mu, kappa)
                # Sample from the distribution
                initial_guess = vmf.draw_samples(size=20, random_state=180)
                initial_guess = normal_vector_to_strike_and_dip(initial_guess)
                return initial_guess

            if len(guess) == 3:
                mu = guess
                # normalise guess
                mu /= np.linalg.norm(mu)
                kappa = 5
                vmf = VonMisesFisher(mu, kappa)
                # Sample from the distribution
                initial_guess = vmf.draw_samples(size=20, random_state=180)
                initial_guess = normal_vector_to_strike_and_dip(initial_guess)
                return initial_guess
            else:
                raise ValueError("'axial_surface_guess' should be a list or a np.array "
                                 "of the form [strike, dip] or a 3D unit vector")

        if 'axial_surface_guess' not in self.kwargs:
            # use the halton method to initialise the optimisation
            return 'halton'

    def setup_optimisation(self, geological_knowledge: Optional[Dict[str, Any]] = None):
        """
           Sets up the optimisation algorithm.

           Parameters
           ----------
           geological_knowledge : dict, optional
               A dictionary containing geological knowledge. Default is None.

           Returns
           -------
           objective_function : callable
               The objective function to be minimised.
           _geological_knowledge : dict or None
               The geological knowledge objective functions.
           solver : BaseOptimiser
               The solver from BaseOptimiser to be used for optimisation.
           guess : Union[np.ndarray, str]
               The initial guess for the optimisation.
        """
        # TODO - check format of the geological knowledge dictionary
        _geological_knowledge, solver = super().setup_optimisation(geological_knowledge)
        # guess = self.generate_initial_guess()

        # Generate initial guess
        guess = self.generate_initial_guess()
        # Setup objective function
        if _geological_knowledge is not None:
            # if _geological_knowledge exists then use the negative logpdf of the Von Mises distribution
            # as the objective function to minimise
            objective_function = self.mle_optimisation()

        # if no geological knowledge is provided then use the angle difference between the predicted and observed
        # foliation as the objective function to minimise
        else:
            objective_function = self.angle_optimisation()

        return objective_function, _geological_knowledge, solver, guess

    def optimise(self, geological_knowledge: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Runs the optimisation.

        Parameters
        ----------
        geological_knowledge : dict, optional
            A dictionary containing geological knowledge. Default is None.

        Returns
        -------
        opt : Dict
            The result of the optimisation.

        Notes
        -----
        This function runs the optimisation by setting up the optimisation problem,
        checking if geological knowledge exists, and running the solver.
        """
        # Setup optimisation
        self.objective_function, self.geo_objective, self.solver, self.guess = \
            self.setup_optimisation(geological_knowledge['fold_axial_surface'])

        # Check if geological knowledge exists
        if geological_knowledge is not None:
            # Check if mode is restricted
            if 'mode' in self.kwargs and self.kwargs['mode'] == 'restricted':
                opt = self.solver(self.objective_function, x0=self.guess, constraints=self.geo_objective)
            else:
                pass
        else:
            opt = self.solver(self.objective_function, x0=self.guess)

        return opt
