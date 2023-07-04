from modified_loopstructural.extra_utils import *
import numpy as np
from scipy.optimize import NonlinearConstraint, BFGS
from LoopStructural.modelling.features.fold import fourier_series
from uncertainty_quantification.fold_uncertainty import *
from knowledge_constraints.splot_processor import SPlotProcessor
from knowledge_constraints._helper import *


def log_likelihood_gaussian(b, mu, sigma):
    return -0.5 * np.log(2 * np.pi * sigma ** 2) - 0.5 * (b - mu) ** 2 / sigma ** 2


class GeologicalKnowledgeConstraints(SPlotProcessor):

    def __init__(self, x, constraints):
        # super().__init__(constraints)
        self.x = x
        self.constraints = constraints
        self.coeff = coeff
        self.constraint_names = ['asymmetry', 'tightness',
                                 'fold_wavelength', 'axial_traces',
                                 'hinge_angle', 'axis_wavelength']

        self.constraint_function_map = None
        self.intercept_function = fourier_series_x_intercepts
        self.splot_function = fourier_series

    def axial_traces(self, theta):
        """
        objective function that evaluates the likelihood of
        fold axial traces given prior knowledge

        Parameters
        ----------
        theta : The fourier series parameters used to calculate the axial traces.

        Returns
        -------
            The likelihood of the axial trace(s).

        """
        intercepts = self.intercept_function(self.x, theta)
        if len(intercepts) == 0:
            return 999
        # print(intercepts)
        else:
            g = 0
            for key, trace in filter(lambda item: "axial_trace" in item[0], self.constraints.items()):
                mu = trace['mu']
                sigma = trace['sigma']
                dist = mu - intercepts
                w = trace['w']
                g += -log_likelihood_gaussian(intercepts[np.argmin(dist)], mu, sigma) * w

        return g

    def wavelength(self, theta):
        """
        Objective function that calculates the likelihood of the wavelength.

        Parameters
        ----------
        theta : np.ndarray
            Fourier series parameters.

        Returns
        -------
            The calculated wavelength.

        Raises
        ------
        TypeError
            If `theta` is not a numpy array or does not have at least 4 elements.
        """
        if not isinstance(theta, np.ndarray):
            raise TypeError("`theta` should be a numpy array.")
        if len(theta) < 4:
            raise ValueError("`theta` should have at least 4 Fourier series parameters.")

        # Get the mu and sigma values from the constraints dictionary
        mu = self.constraints['fold_wavelength']['mu']
        sigma = self.constraints['fold_wavelength']['sigma']
        # Calculate the likelihood
        likelihood = -log_likelihood_gaussian(theta[3], mu, sigma)
        # Get the weight of the constraint
        w = self.constraints['fold_wavelength']['w']
        # Multiply the likelihood by the weight
        likelihood *= w

        return likelihood

    def axis_wavelength(self, theta):
        """
        objective function that evaluates the likelihood of
        the fold axis wavelength given prior knowledge

        Parameters
        ----------
        theta : The fourier series parameters used to calculate the axial traces.

        Returns
        -------
            The likelihood of the axial trace(s).

        """
        mu = self.constraints['axis_wavelength']['mu']
        sigma = self.constraints['axis_wavelength']['sigma']
        g = -log_likelihood_gaussian(theta[3], mu, sigma)
        w = self.constraints['axis_wavelength']['w']
        g *= w

        return g

    def tightness(self, theta):
        mu = self.constraints['tightness']['mu']
        sigma = self.constraints['tightness']['sigma']
        g = -log_likelihood_gaussian(self.calculate_tightness(theta), mu, sigma)
        w = self.constraints['tightness']['w']
        g *= w

        return g

    def hinge_angle(self, theta):
        mu = self.constraints['hinge_angle']['mu']
        sigma = self.constraints['hinge_angle']['sigma']
        g = -log_likelihood_gaussian(self.calculate_tightness(theta), mu, sigma)
        w = self.constraints['hinge_angle']['w']
        g *= w

        return g

    def asymmetry(self, theta):
        mu = self.constraints['asymmetry']['mu']
        sigma = self.constraints['asymmetry']['sigma']
        g = -log_likelihood_gaussian(self.calculate_asymmetry(theta), mu, sigma)
        w = self.constraints['asymmetry']['w']
        g *= w

        return g

    def setup_constraint_functions(self):
        self.constraint_function_map = {'asymmetry': self.asymmetry,
                                        'tightness': self.tightness,
                                        'fold_wavelength': self.wavelength,
                                        'axis_wavelength': self.axis_wavelength,
                                        'axial_traces': self.axial_traces,
                                        'hinge_angle': self.hinge_angle
                                        }

    def __call__(self, theta):

        self.setup_constraint_functions()

        g = 0
        for key in self.constraint_names:

            if key in self.constraints:
                g += self.constraint_function_map[key](theta)

            else:
                pass

        return g

    def prepare_and_setup_constraints(self):
        self.setup_constraint_functions()

        # feaisble_interval = {}
        constraints = []
        for constraint_name, constraint_info in self.constraints.items():
            lb = constraint_info['lb']
            ub = constraint_info['ub']
            mu = constraint_info['mu']
            sigma = constraint_info['sigma']

            val = -log_likelihood_gaussian(np.linspace(lb, ub, 100), mu, sigma)
            nlc = NonlinearConstraint(self.constraint_function_map[constraint_name],
                                      val.min(), val.max(),
                                      jac='2-point', hess=BFGS())
            constraints.append(nlc)

        return constraints
