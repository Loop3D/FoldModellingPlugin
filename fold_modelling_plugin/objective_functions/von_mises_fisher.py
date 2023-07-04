from scipy.stats import vonmises_fisher
import numpy as np


class VonMisesFisher:
    """
    Class defines objective functions based on Von MisesFisher (vMF) distribution
    to be used in MLE optimisation problems, and drawing samples from vMF distribution.

    """

    def __init__(self, mu, kappa):
        """
        initialise the von Mises-Fisher distribution class.

        Parameters
        ----------
        mu : Mean direction vector, in the form [mu_x, mu_y, mu_z].
        kappa : Concentration parameter (always positive).

        Raises
        ------
        ValueError
            If `mu` is not a three-element vector or `kappa` is not a positive number.
        """
        if len(mu) != 3:
            raise ValueError("`mu` should be a three-element vector.")
        if kappa <= 0:
            raise ValueError("`kappa` should be a positive number.")

        # initialise the the Von Mises Fisher distribution
        self.vmf = vonmises_fisher(mu, kappa)

    def pdf(self, x):
        """
        Evaluate the PDF of the von Mises-Fisher distribution at a given point.

        Parameters
        ----------
        x : np.ndarray
            Points at which the PDF should be evaluated.

        Returns
        -------
        np.ndarray
            The evaluated PDF.
        """
        pdf = self.vmf.pdf(x)

        return pdf

    def logpdf(self, x):
        """
        Evaluate the log of the PDF of the von Mises-Fisher distribution at a given point.

        Parameters
        ----------
        x : np.ndarray
            Points at which the log of the PDF should be evaluated.

        Returns
        -------
        np.ndarray
            The evaluated log of the PDF.

        """

        logpdf = self.vmf.logpdf(x)

        return logpdf

    def draw_samples(self, size=1, random_state=1):
        """
        Draw random samples from the von Mises-Fisher distribution.

        Parameters
        ----------
        size : int, optional
            Number of samples to draw, by default 1.
        random_state : int, optional
            Seed for the random number generator, by default 1.

        Returns
        -------
        np.ndarray
            The drawn samples.

        Raises
        ------
        TypeError
            If `size` or `random_state` is not an integer.
        """
        if not isinstance(size, int):
            raise TypeError("`size` should be an integer.")
        if not isinstance(random_state, int):
            raise TypeError("`random_state` should be an integer.")

        random_samples = self.vmf.rvs(size=size, random_state=random_state)

        return random_samples
