import numpy
from scipy.optimize import NonlinearConstraint, BFGS

# from LoopStructural.modelling.features.fold import fourier_series
from typing import Union, Dict, List
from ..helper.utils import *
from ..splot.splot_processor import SPlotProcessor
from .von_mises_fisher import VonMisesFisher
from .objective_functions import ObjectiveFunction
from ..datatypes import ObjectiveType, KnowledgeType, InputGeologicalKnowledge, FitType
import beartype


def check_fourier_parameters(theta):
    if not isinstance(theta, numpy.ndarray):
        raise TypeError("`theta` should be a numpy array.")
    if len(theta) < 4:
        raise ValueError("`theta` should have at least 4 Fourier series parameters.")


@beartype.beartype
class GeologicalKnowledgeFunctions(SPlotProcessor):
    """
    Base class for geological knowledge objective functions

    """

    def __init__(self, input_knowledge: InputGeologicalKnowledge):
        # TODO add attribute to use any custom function otherwise use gaussian likelihood
        """
        Initialize the GeologicalKnowledgeConstraints class.

        Parameters
        ----------
        x : np.ndarray
            The values of the fold frame coordinates (0 or 1) used to calculate the fitted fold rotation angle curve.

        input_knowledge : InputGeologicalKnowledge
            The input geological knowledge constraints.

        """

        # Initialise the x values, constraints
        SPlotProcessor.__init__(self)
        # self.x = None
        self.input_knowledge = input_knowledge

        # Define the constraint names
        self.constraint_names = [
            "asymmetry",
            "tightness",
            "fold_wavelength",
            "axial_trace",
            "hinge_angle",
            "fold_axis_wavelength",
            "axial_surface",
        ]

        # Initialise the objective function map
        self.objective_functions_map = self.create_objective_functions_map()

        # Define the intercept function and splot function
        self.intercept_function = fourier_series_x_intercepts
        self.splot_function = fourier_series
        self.fittypeflag = [False] * len(FitType)

    def axial_surface_objective_function(self, vector: Union[List, numpy.ndarray]) -> float:
        """
        Objective function for the axial surface.
        This function calculates the loglikelihood of an axial surface using the VonMisesFisher distribution.

        Parameters
        ----------
        vector : numpy.ndarray
            the unit vector that represents the axial surface.
            This is evaluated every iteration in the optimisation process

        Returns
        -------
        float
            The logpdf value from the VonMisesFisher distribution.
        """
        if len(vector) != 3:
            raise ValueError("The input array or list should be a 3D vector of type e.g., [0.0, 0.0, 0.0]")

        # Extract parameters for the VonMisesFisher distribution
        mu = self.input_knowledge[KnowledgeType.AXIAL_SURFACE].mu
        kappa = self.input_knowledge[KnowledgeType.AXIAL_SURFACE].kappa
        # Create a VonMisesFisher distribution with the given parameters mu and kappa
        vmf = VonMisesFisher(mu, kappa)

        # Calculate the logpdf of the input array
        vmf_logpdf = (
            vmf.logpdf(vector)
            * self.input_knowledge[KnowledgeType.AXIAL_SURFACE].weight
        )

        return vmf_logpdf

    def axial_trace_objective_function(self, theta: Union[List, numpy.ndarray]) -> Union[int, float]:

        """
        Calculate the objective function for the 'axial_trace' constraint.
        This function calculates the negative likelihood of the axial trace(s) given the provided knowledge constraints.

        Parameters
        ----------
        theta : np.ndarray
            The Fourier series parameters. These are the parameters of the Fourier series used to calculate
            the axial traces.

        Returns
        -------
        Union[int, float]
            The calculated objective function value. This is the likelihood of the axial trace(s).
            If there are no axial traces, the function returns a predefined constant value (999) that
             penalise the minimisation algorithm.
        """
        # Check if theta is an array and has at least 4 parameters
        # If not, an exception will be raised
        check_fourier_parameters(theta)

        # Calculate the intercepts using the provided theta values and the x values of the class
        intercepts = self.intercept_function(self.x, theta)
        if len(intercepts) == 0:
            # If there are no intercepts, return the predefined constant value
            return 999
        else:
            # Initialize the likelihood to 0
            likelihood = 0
            # Iterate over the knowledge constraints dictionary that have "axial_trace" in their key
            for key, trace in filter(
                lambda item: "axial_trace" in item[0],
                self.input_knowledge[KnowledgeType.AXIAL_TRACE].items(),
            ):
                # Get the mu, sigma, and weight values of a given axial trace
                # These values represent the mean and standard deviation of the location of a given axial trace
                mu = trace.mu
                sigma = trace.sigma
                w = trace.weight
                # Calculate the distance between mu and the axial trace
                dist = mu - intercepts
                # Update the likelihood using the Gaussian log likelihood function
                likelihood += (
                    -ObjectiveFunction[ObjectiveType.LOG_NORMAL](
                        intercepts[np.argmin(dist)], mu, sigma
                    )
                    * w
                )

        return likelihood

    def wavelength_objective_function(self, theta: numpy.ndarray) -> float:
        """
        Calculate the objective function for the fold wavelength constraint.

        This function calculates the negative likelihood of the fold wavelength given the knowledge constraints.

        Parameters
        ----------
        theta : np.ndarray
            The Fourier series parameters. These are the parameters of the Fourier series used to
            calculate the fold limb rotation angle curve.

        Returns
        -------
        float
            The calculated objective function value. This is the likelihood of the fold wavelength.

        Raises
        ------
        TypeError
            If `theta` is not a numpy array.
        ValueError
            If `theta` does not have at least 4 parameters.
        """
        # Check if theta is an array and has at least 4 parameters
        # If not, an exception will be raised
        check_fourier_parameters(theta)

        # Get the mu and sigma values
        # These values represent the mean and standard deviation of the fold wavelength
        mu = self.input_knowledge[KnowledgeType.WAVELENGTH].mu
        sigma = self.input_knowledge[KnowledgeType.WAVELENGTH].sigma
        # Calculate the likelihood of the fold wavelength
        # The likelihood is calculated using the Gaussian log likelihood function
        likelihood = -ObjectiveFunction[ObjectiveType.LOG_NORMAL](theta[3], mu, sigma)

        # The weight is used to adjust the influence of this constraint on the overall objective function
        # Multiply the likelihood by the weight to get the final objective function value
        likelihood *= self.input_knowledge[KnowledgeType.WAVELENGTH].weight

        return likelihood

    def fold_axis_wavelength_objective_function(self, theta: numpy.ndarray) -> float:
        """
        Calculate the objective function for the fold axis wavelength constraint.

        This function calculates the negative likelihood of the fold axis wavelength given the constraints.

        Parameters
        ----------
        theta : np.ndarray
            The Fourier series parameters. These are the parameters of the Fourier series used to
            calculate the fold axis rotation angle curve.

        Returns
        -------
        float
            The calculated objective function value. This is the likelihood of the fold axis wavelength.

        Raises
        ------
        TypeError
            If `theta` is not a numpy array.
        ValueError
            If `theta` does not have at least 4 parameters.
        """
        # Check if theta is an array and has at least 4 parameters
        # If not, an exception will be raised
        check_fourier_parameters(theta)

        # Get the mu and sigma values from the constraints dictionary
        # These values represent the mean and standard deviation of the fold axis wavelength
        mu = self.input_knowledge[KnowledgeType.AXIS_WAVELENGTH].mu
        sigma = self.input_knowledge[KnowledgeType.AXIS_WAVELENGTH].sigma
        # Calculate the likelihood of the fold axis rotation angle wavelength
        # The likelihood is calculated using the negative Gaussian log likelihood function
        likelihood = -ObjectiveFunction[ObjectiveType.LOG_NORMAL](theta[3], mu, sigma)

        # The weight is used to adjust the influence of this constraint on the overall objective function
        # Multiply the likelihood by the weight to get the final objective function value
        likelihood *= self.input_knowledge[KnowledgeType.AXIS_WAVELENGTH].weight

        return likelihood

    def tightness_objective_function(self, theta: numpy.ndarray) -> float:
        """
        Calculate the objective function for the 'tightness' constraint.

        This function calculates the likelihood of the fold tightness given the constraints.

        Parameters
        ----------
        theta : np.ndarray
            The Fourier series parameters. These are the parameters of the Fourier series used to
            calculate the fold limb rotation angle curve.

        Returns
        -------
        float
            The calculated objective function value. This is the likelihood of the fold tightness.

        Raises
        ------
        TypeError
            If `theta` is not a numpy array.
        ValueError
            If `theta` does not have at least 4 parameters.
        """
        # Check if theta is an array and has at least 4 parameters
        # If not, an exception will be raised
        check_fourier_parameters(theta)

        # Get the mu, sigma, and weight values from the constraints dictionary
        # These values represent the mean, standard deviation, and weight of the fold tightness
        mu = self.input_knowledge[KnowledgeType.TIGHTNESS].mu
        sigma = self.input_knowledge[KnowledgeType.TIGHTNESS].sigma
        # Calculate the tightness of the fold
        tightness = self.calculate_tightness(theta)

        # Calculate the likelihood of the fold tightness
        # The likelihood is calculated using the negative Gaussian log likelihood function
        likelihood = -ObjectiveFunction[ObjectiveType.LOG_NORMAL](tightness, mu, sigma)

        # Multiply the likelihood by the weight to get the final objective function value
        likelihood *= self.input_knowledge[KnowledgeType.TIGHTNESS].weight

        return likelihood

    def hinge_angle_objective_function(self, theta: numpy.ndarray) -> float:
        """
        Calculate the objective function for the 'hinge_angle' constraint.

        This function calculates the likelihood of the fold hinge angle given the constraints.

        Parameters
        ----------
        theta : np.ndarray
            The Fourier series parameters. These are the parameters of the Fourier series used
            to calculate the fold axis rotation angle curve.

        Returns
        -------
        float
            The calculated objective function value. This is the likelihood of the fold hinge angle.

        Raises
        ------
        TypeError
            If `theta` is not a numpy array.
        ValueError
            If `theta` does not have at least 4 parameters.
        """
        # Check if theta is an array and has at least 4 parameters
        # If not, an exception will be raised
        check_fourier_parameters(theta)

        # Get the mu, sigma, and weight values from the constraints dictionary
        # These values represent the mean, standard deviation, and weight of the fold hinge angle
        # Calculate the hinge angle of the fold
        hinge_angle = self.calculate_tightness(theta)

        # Calculate the likelihood of the fold hinge angle
        # The likelihood is calculated using the negative Gaussian log likelihood function
        likelihood = -ObjectiveFunction[ObjectiveType.LOG_NORMAL](
            hinge_angle,
            self.input_knowledge[KnowledgeType.HINGE_ANGLE].mu,
            self.input_knowledge[KnowledgeType.HINGE_ANGLE].sigma,
        )

        # Multiply the likelihood by the weight to get the final objective function value
        likelihood *= self.input_knowledge[KnowledgeType.HINGE_ANGLE].weight

        return likelihood

    def asymmetry_objective_function(self, theta: numpy.ndarray) -> float:
        """
        Calculate the objective function for the 'asymmetry' constraint.

        This function calculates the likelihood of the fold asymmetry degree given the constraints.

        Parameters
        ----------
        theta : np.ndarray
            The Fourier series parameters. These are the parameters of the Fourier series used to
            calculate the fold limb rotation angle curve.

        Returns
        -------
        float
            The calculated objective function value. This is the likelihood of the fold asymmetry degree.

        Raises
        ------
        TypeError
            If `theta` is not a numpy array.
        ValueError
            If `theta` does not have at least 4 parameters.
        """
        # Check if theta is an array and has at least 4 parameters
        # If not, an exception will be raised
        check_fourier_parameters(theta)

        # Get the mu, sigma, and weight values from the constraints dictionary
        # These values represent the mean, standard deviation, and weight of the fold asymmetry degree
        mu = self.input_knowledge[KnowledgeType.ASYMMETRY].mu
        sigma = self.input_knowledge[KnowledgeType.ASYMMETRY].sigma
        # Calculate the asymmetry of the fold
        asymmetry = self.calculate_asymmetry(theta)

        # Calculate the likelihood of the fold asymmetry
        # The likelihood is calculated using the negative Gaussian log likelihood function
        likelihood = -ObjectiveFunction[ObjectiveType.LOG_NORMAL](asymmetry, mu, sigma)

        # Multiply the likelihood by the weight to get the final objective function value
        likelihood *= self.input_knowledge[KnowledgeType.ASYMMETRY].weight

        return likelihood

    def create_objective_functions_map(self):
        """
        Setup the mapping between constraint names and their corresponding objective function methods.

        This function creates a dictionary where the keys are the names of the constraints and the values are the
        corresponding objective function methods. This mapping is used to dynamically call the correct objective
        function based on the constraint name.
        """
        # Create a dictionary to map the constraint names to their corresponding objective function methods
        objective_functions_map = {
            KnowledgeType.ASYMMETRY: self.asymmetry_objective_function,
            KnowledgeType.TIGHTNESS: self.tightness_objective_function,
            KnowledgeType.WAVELENGTH: self.wavelength_objective_function,
            KnowledgeType.AXIS_WAVELENGTH: self.fold_axis_wavelength_objective_function,
            KnowledgeType.AXIAL_TRACE: self.axial_trace_objective_function,
            KnowledgeType.HINGE_ANGLE: self.hinge_angle_objective_function,
            KnowledgeType.AXIAL_SURFACE: self.axial_surface_objective_function,
        }

        return objective_functions_map

    def setup_objective_functions_for_restricted_mode(
        self,
    ) -> List[NonlinearConstraint]:
        """

        This function prepares the constraints by calculating the lower and upper bounds for each constraint and
        setting up the NonlinearConstraint objects from scipy.optimize. It also sets up the constraint functions.

        This function is used only when the optimisation is in a restricted mode. The restricted mode means that the
        optimisation algorithm cannot leave the parameter space defined by the geological knowledge constraints.
        The drawback of fitting a fold rotation angle model in this mode is that if the constraints provided are not
        representative of the studied fold geometry, the fitted model will be as well not representative.

        Returns
        -------
        List[NonlinearConstraint]
            A list of NonlinearConstraint objects for each constraint.

        Raises
        ------
        TypeError
            If the constraint info is not a dictionary.
        """

        # Initialize the list of constraints
        constraints = []
        # Iterate over all constraints
        for constraint_name, constraint_info in self.input_knowledge.items():
            # Check if constraint_info is a dictionary
            if not isinstance(constraint_info, dict):
                raise TypeError("`constraint_info` should be a dictionary.")

            # Get the lower and upper bounds, mu, and sigma values from the constraint info
            lb = constraint_info["lb"]
            ub = constraint_info["ub"]
            mu = constraint_info["mu"]
            sigma = constraint_info["sigma"]

            # Calculate the negative Gaussian log likelihood for a range of values between the lower and upper bounds
            val = -ObjectiveFunction[ObjectiveType.LOG_NORMAL](
                np.linspace(lb, ub, 100), mu, sigma
            )
            # Create a NonlinearConstraint object for this constraint
            nlc = NonlinearConstraint(
                self.objective_functions_map[constraint_name],
                val.min(),
                val.max(),
                jac="2-point",
                hess=BFGS(),
            )
            # Add the NonlinearConstraint object to the list of constraints
            constraints.append(nlc)

        # Return the list of constraints
        return constraints

    def __call__(self, theta: numpy.ndarray) -> float:
        """
        Calculate the total geological knowledge objective function value for all constraints by summing up the
        objective function values for all constraints. This objective function represent only the
        knowledge constraints and it is minimised with the main objective function that calculates the residuals.

        Parameters
        ----------
        theta : np.ndarray
            The Fourier series parameters 1x4 or unit vector 1x3. These are the parameters of the Fourier series used to
            calculate the fold limb and axis rotation angle curve.

        Returns
        -------
        float
            The total objective function value.

        Raises
        ------
        TypeError
            If `theta` is not a numpy array.
        """

        # Initialize the total objective function value to 0
        total_objective_value = 0
        if (
            len(theta) == 3
            and self.input_knowledge[KnowledgeType.AXIAL_SURFACE] is not None
        ):
            total_objective_value += self.axial_surface_objective_function(theta)

        elif len(theta) == 4:
            if self.fittypeflag[FitType.LIMB] is True:
                if self.input_knowledge[KnowledgeType.WAVELENGTH] is not None:
                    total_objective_value += self.wavelength_objective_function(theta)

                elif self.input_knowledge[KnowledgeType.ASYMMETRY] is not None:
                    total_objective_value += self.asymmetry_objective_function(theta)

                elif self.input_knowledge[KnowledgeType.TIGHTNESS] is not None:
                    total_objective_value += self.tightness_objective_function(theta)

                elif self.input_knowledge[KnowledgeType.AXIAL_TRACE] is not None:
                    total_objective_value += self.axial_trace_objective_function(theta)

                # update the flag
                self.fittypeflag[FitType.LIMB] = False

            elif self.fittypeflag[FitType.AXIS] is True:
                if self.input_knowledge[KnowledgeType.HINGE_ANGLE] is not None:
                    total_objective_value += self.hinge_angle_objective_function(theta)

                elif self.input_knowledge[KnowledgeType.AXIS_WAVELENGTH] is not None:
                    total_objective_value += (
                        self.fold_axis_wavelength_objective_function(theta)
                    )

                # update the flag
                self.fittypeflag[FitType.AXIS] = False

            else:
                total_objective_value += 0.
        else: 
            raise ValueError("The input array or list should be a 3D vector of type e.g., [0.0, 0.0, 0.0] or 4 Fourier Series parameters.")

            # Return the total objective function value
        return total_objective_value

    def __getitem__(self, knowledge_type: KnowledgeType):
        """
        Get the geological knowledge constraints for a given knowledge type.

        Parameters
        ----------
        knowledge_type : KnowledgeType
            The knowledge type.

        Returns
        -------
        Union[NormalDistribution, VonMisesFisherDistribution]
            The knowledge constraints for the given knowledge type.
        """
        return self.input_knowledge(knowledge_type)




