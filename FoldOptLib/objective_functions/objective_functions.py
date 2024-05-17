from ..datatypes.enums import ObjectiveType
import numpy
from typing import Union, List
from ..utils.utils import get_predicted_rotation_angle
from scipy.stats import vonmises
import beartype


class MetaObjectiveFunction(type):
    def __getitem__(cls, item):
        return cls.map_functions()[item]


class ObjectiveFunction(metaclass=MetaObjectiveFunction):
    """
    This class contains the functions to calculate the log-likelihood of a Gaussian distribution and the VonMisesFisher
    distribution.

    """

    # @beartype.beartype
    @staticmethod
    def log_normal(
        b: Union[int, float], mu: Union[int, float], sigma: Union[int, float] = 1e-2
    ):
        """
        Calculate the log-likelihood of a Gaussian distribution.

        This function calculates the log-likelihood of a Gaussian distribution for a given observed value, mean,
        and standard deviation. The formula used is the standard formula for the log-likelihood of a Gaussian
        distribution.

        Parameters
        ----------
        b : Union[int, float]
            The observed value.
        mu : Union[int, float]
            The mean of the Gaussian distribution.
        sigma : Union[int, float]
            The standard deviation of the Gaussian distribution. by default is 1e-2.

        Returns
        -------
        float
            The calculated log-likelihood.

        Raises
        ------
        ValueError
            If `sigma` is less than or equal to 0.
        """

        # Check if sigma is greater than 0
        if sigma <= 0:
            raise ValueError("`sigma` should be greater than 0.")

        # Calculate the log-likelihood
        likelihood = (
            -0.5 * numpy.log(2 * numpy.pi * sigma**2) - 0.5 * (b - mu) ** 2 / sigma**2
        )

        # Return the log-likelihood
        return -likelihood

    @beartype.beartype
    @staticmethod
    def vonmises(x: Union[list, numpy.ndarray]):
        """
        Objective function for the axial surface.
        This function calculates the loglikelihood of an axial surface using the VonMisesFisher distribution.

        Parameters
        ----------
        x : float
            represents the angle between the observed folded foliation and the predicted one.

        Returns
        -------
        Union[int, float]
            The logpdf value from the VonMises distribution.
        """
        # Define the mu and kappa of the VonMises distribution
        # mu = 0 because we want to minimise the angle between the observed and predicted folded foliation
        # kappa = 100 because we want to have a sharp distribution very close to the mean 0 (mu)
        mu = 1e-10
        kappa = 100

        # Create a VonMises distribution with the given parameters
        vm = vonmises(mu, kappa)

        # Calculate the logpdf of the input array
        vm_logpdf = -vm.logpdf(x)

        if isinstance(vm_logpdf, numpy.ndarray):
            vm_logpdf = vm_logpdf.sum()
            return vm_logpdf
        else:
            return vm_logpdf

    @beartype.beartype
    @staticmethod
    def fourier_series(
        rotation_angle: numpy.ndarray, fold_frame_coordinate: numpy.ndarray
    ):
        """
        Wrapper function to calculate the log-likelihood of a Fourier series.
        Args
        ----------
        rotation_angle (numpy.ndarray): The observed rotation angle.
        fold_frame_coordinate (numpy.ndarray): The fold frame coordinate of the folded foliation.

        Returns
        -------

        """

        def objective_fourier_series(theta):
            y = rotation_angle
            y_pred = get_predicted_rotation_angle(theta, fold_frame_coordinate)
            log_likelihood = 0
            for fr, fd in zip(y, y_pred):
                log_likelihood += ObjectiveFunction.log_normal(fr, fd)

            return log_likelihood

        return objective_fourier_series

    @beartype.beartype
    @staticmethod
    def angle_function(v1: Union[List, numpy.ndarray], v2: Union[List, numpy.ndarray]):
        """
        Calculate the angle difference between the predicted bedding and the observed one.

        This is an objective function that verifies if the predicted and observed bedding or folded foliation are
        geometrically compatible. If the sum of the calculated angles is close to 0 degrees, the axial plane of the
        predicted folded foliation is representative of the axial surface of observed folded foliation.

        Parameters
        ----------
        v1 : np.ndarray
            The first vector representing the predicted bedding.
        v2 : np.ndarray
            The second vector representing the observed bedding.

        Returns
        -------
        np.ndarray
            The angle difference between the predicted and observed bedding.

        Raises
        ------
        ValueError
            If `v1` and `v2` are not numpy arrays of the same shape.
        """
        # Check if `v1` and `v2` are numpy arrays of the same shape
        if not isinstance(v1, numpy.ndarray) or not isinstance(v2, numpy.ndarray):
            raise ValueError("`v1` and `v2` should be numpy arrays.")

        if v1.shape != v2.shape:
            raise ValueError("`v1` and `v2` should have the same shape.")

        # Calculate the dot product of `v1` and `v2`
        dot_product = numpy.einsum("ij,ij->i", v1, v2)
        # Calculate the angle between v1 and v2
        angle_difference = numpy.arccos(dot_product)

        total_angle_difference = angle_difference.sum()

        return total_angle_difference

    @staticmethod
    def map_functions():

        return {
            ObjectiveType.LOG_NORMAL: ObjectiveFunction.log_normal,
            ObjectiveType.VON_MISES: ObjectiveFunction.vonmises,
            ObjectiveType.FOURIER: ObjectiveFunction.fourier_series,
            ObjectiveType.ANGLE: ObjectiveFunction.angle_function,
        }
