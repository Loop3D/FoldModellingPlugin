from modified_loopstructural.extra_utils import *
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, normalize
import numpy as np
from LoopStructural.modelling.features.fold import fourier_series
from uncertainty_quantification.fold_uncertainty import *
# from _helper import *
from scipy.optimize import minimize, differential_evolution
from fold_modelling_plugin.objective_functions.geological_knowledge import GeologicalKnowledgeFunctions
from knowledge_constraints.splot_processor import SPlotProcessor
from knowledge_constraints._helper import *
from fold_optimiser import FoldOptimiser
from fold_modelling_plugin.objective_functions.gaussian import loglikelihood_fourier_series
from fold_modelling_plugin.utils import calculate_semivariogram, fourier_series


def scale(data):
    mms = MinMaxScaler()
    mms.fit(data)
    data_transformed = mms.transform(data)

    return data_transformed


def get_wavelength_guesses(guess, size):
    np.random.seed(180)
    mu, sigma = guess, guess / 3
    return np.random.normal(mu, abs(sigma), size)

def objective_wrapper(func1, func2):
    def objective_function(x):
        return func1(x) + func2(x)
    return objective_function


class FourierSeriesOptimiser(FoldOptimiser):

    def __init__(self, fold_frame_coordinate, rotation_angle,
                 knowledge_constraints=None, x, **kwargs):

        self.objective_value = 0
        self.fold_frame_coordinate = fold_frame_coordinate
        self.rotation_angle = np.tan(np.deg2rad(rotation_angle))
        # TODO: Add a check if the knowledge constraints are in the correct format
        self.knowledge_constraints = knowledge_constraints
        self.x = x
        self.kwargs = kwargs

    def prepare_and_setup_knowledge_constraints(self):
        """
        Prepare the geological knowledge constraints and objective functions
        """

        if self.knowledge_constraints is not None:
            if 'mode' in self.kwargs and self.kwargs['mode'] == 'restricted':
                geological_knowledge = GeologicalKnowledgeFunctions(self.x, self.knowledge_constraints)
                ready_constraints = geological_knowledge.setup_objective_functions_for_restricted_mode(self)

                return ready_constraints

            else:

                geological_knowledge = GeologicalKnowledgeFunctions(self.x, self.knowledge_constraints)

                return geological_knowledge

        if self.knowledge_constraints is None:
            return None

    def setup_optimisation(self):
        """
        Setup fourier series optimisation
        """

        if 'method' in self.kwargs:
            if self.kwargs['method'] == 'differential_evolution':
                solver = self.optimise_with_differential_evolution

            if self.kwargs['method'] == 'trust-constr':
                solver = self.optimise_with_trust_region

        if 'method' not in self.kwargs:
            solver = self.optimise_with_trust_region

        objective_function = loglikelihood_fourier_series(self.rotation_angle,
                                                          self.fold_frame_coordinate)

        geological_knowledge = self.prepare_and_setup_knowledge_constraints()
        guess = self.generate_initial_guess()

        return objective_function, geological_knowledge, solver, guess

    def generate_initial_guess(self):
        """
        Generate an initial guess for the optimisation
        It generates a guess of the wavelength, for fourier series. the format of the guess depends
        if the method of optimisation is differential evolution or trust region. If it's differential evolution
        it will generate the bounds for the optimisation, if it's trust region it will generate the initial guess of
        the wavelength

        """
        if 'method' in self.kwargs:
            if self.kwargs['method'] == 'differential_evolution':
                if 'wl_guess' in self.kwargs:
                    wl = get_wavelength_guesses(self.kwargs['wl_guess'], 1000)
                else:
                    # calculate semivariogram and get the wavelength guess
                    guess, lags, variogram = calculate_semivariogram(self.rotation_angle,
                                                                     self.fold_frame_coordinate)
                    wl = get_wavelength_guesses(x0[3], 1000)
                    bounds = np.array([(-1, 1), (-1, 1), (-1, 1),
                                       (wl[wl > 0].min() / 2, wl.max())], dtype=object)
                    return bounds

        if 'wl_guess' in self.kwargs:
            guess = np.array([0, 1, 1, self.kwargs['wl_guess']], dtype=object)
        else:
            # calculate semivariogram and get the wavelength guess
            guess, lags, variogram = calculate_semivariogram(self.rotation_angle,
                                                             self.fold_frame_coordinate)

            return guess

    def optimise(self):
        """
        Optimise the fourier series
        """
        objective_function, geological_knowledge, solver, guess = self.setup_optimisation()

        if geological_knowledge is not None:
            if 'mode' in self.kwargs and self.kwargs['mode'] == 'restricted':
                opt = solver(objective_function, x0=guess, constraints=geological_knowledge)
            else:
                objective_function = objective_wrapper(objective_function, geological_knowledge)
                opt = solver(objective_function, x0=guess, constraints=geological_knowledge.constraints)

        else:
            opt = solver(objective_function, x0=guess)

        return opt

    def __call__(self, *args, **kwargs):

    def fit_constrained_fourier_series(self, w, x0=None):

        # constraints = self.knowledge_constraints.prepare_and_setup_constraints()

        if x0 is not None:
            opt = minimize(self.mle_objective, x0,
                           # constraints=constraints,
                           method='trust-constr', jac='2-point')

        if self.coeff == 4:
            x0 = np.array([0, 1, 1, w], dtype=object)

        if self.coeff == 8:
            x0 = np.array([0, 1, 1, w[0], 1, 1, w[1]], dtype=object)

        opt = minimize(self.mle_objective, x0,
                       # constraints=constraints,
                       method='trust-constr', jac='2-point')

        return opt

    def fit_constrained_fourier_series_DE(self, wls, x0=None):

        # constraints = self.knowledge_constraints.prepare_and_setup_constraints()

        # bounds = np.array([(-1, 1), (-1, 1), (-1, 1), (x0[0]/2, wls.max())])

        if self.coeff == 4:
            # x0 = np.array([0, 1, 1, wls])
            wl = get_wavelength_guesses(x0[3], 1000)
            bounds = np.array([(-1, 1), (-1, 1), (-1, 1),
                               (wl[wl > 0].min() / 2, wl.max())], dtype=object)

        opt = differential_evolution(self.mle_objective,
                                     bounds=bounds,
                                     init='halton',
                                     maxiter=5000,
                                     # seed=80,
                                     polish=True,
                                     strategy='best2exp',
                                     mutation=(0.3, 0.99),
                                     )

        return opt
