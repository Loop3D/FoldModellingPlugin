from ..datatypes.enums import LikelihoodType
import numpy
from typing import Union
from ..helper.utils import get_predicted_rotation_angle
from scipy.stats import vonmises
import beartype


class LikelihoodFunction:
    """
    This class contains the functions to calculate the log-likelihood of a Gaussian distribution and the VonMisesFisher
    distribution.

    """
    def __init__(self):
        self.functions_map = self.map_functions()

    @beartype.beartype
    @staticmethod
    def gaussian_log_likelihood(
            b: Union[int, float],
            mu: Union[int, float],
            sigma: Union[int, float] = 1e-2
    ) -> float:
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
        likelihood = -0.5 * numpy.log(2 * numpy.pi * sigma ** 2) - 0.5 * (b - mu) ** 2 / sigma ** 2

        # Return the log-likelihood
        return -likelihood

    @beartype.beartype
    @staticmethod
    def loglikelihood_axial_surface(x: Union[list, numpy.ndarray]) -> Union[int, float]:
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
    def loglikelihood_fourier_series(rotation_angle: numpy.ndarray, fold_frame_coordinate: numpy.ndarray):
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
                log_likelihood += LogLikelihood.gaussian_log_likelihood(fr, fd)

            return log_likelihood

        return objective_fourier_series

    @staticmethod
    def map_functions():

        return {
            LikelihoodType.LogNormal: LikelihoodFunction.gaussian_log_likelihood,
            LikelihoodType.VonMisesFisher: LikelihoodFunction.loglikelihood_axial_surface,
            LikelihoodType.FourierSeries: LikelihoodFunction.loglikelihood_fourier_series
        }

    @beartype.beartype
    def __call__(self, likelihood_type: LikelihoodType):

        return self.functions_map[likelihood_type]
