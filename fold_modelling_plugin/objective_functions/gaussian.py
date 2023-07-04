import numpy as np
from typing import Union


def gaussian_log_likelihood(b: Union[int, float], mu: Union[int, float], sigma: Union[int, float]) -> float:
    """
    Calculate the log-likelihood of a Gaussian distribution.

    This function calculates the log-likelihood of a Gaussian distribution for a given observed value, mean,
    and standard deviation. The formula used is the standard formula for the log-likelihood of a Gaussian distribution.

    Parameters
    ----------
    b : Union[int, float]
        The observed value.
    mu : Union[int, float]
        The mean of the Gaussian distribution.
    sigma : Union[int, float]
        The standard deviation of the Gaussian distribution.

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
    likelihood = -0.5 * np.log(2 * np.pi * sigma ** 2) - 0.5 * (b - mu) ** 2 / sigma ** 2

    # Return the log-likelihood
    return likelihood
