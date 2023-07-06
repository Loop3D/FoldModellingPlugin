from modified_loopstructural.extra_utils import *
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, normalize
import numpy as np
from LoopStructural.modelling.features.fold import fourier_series
from uncertainty_quantification.fold_uncertainty import *
# from _helper import *
from scipy.optimize import minimize, differential_evolution
from knowledge_constraints.knowledge_constraints import GeologicalKnowledgeConstraints
from knowledge_constraints.splot_processor import SPlotProcessor
from knowledge_constraints._helper import *


def scale(data):
    mms = MinMaxScaler()
    mms.fit(data)
    data_transformed = mms.transform(data)

    return data_transformed


def get_wavelength_guesses(guess, size):
    np.random.seed(180)
    mu, sigma = guess, guess / 3
    return np.random.normal(mu, abs(sigma), size)


class FourierSeriesOptimiser:

    def __init__(self, fold_frame, rotation_angle, constraints,
                 x, n_traces=None, at_constrain_only=None,
                 coeff=4, w=1.):
        # self.objective_function = objective_function
        # super().__init__( objective, fold_frame, rotation_angle, constraints,
        #          x, n_traces=None, at_constrain_only=None,
        #          coeff=4)
        self.objective_value = 0
        self.fold_frame = fold_frame
        self.rotation_angle = np.tan(np.deg2rad(rotation_angle))
        self.constraints = constraints
        self.x = x
        # self.y = np.tan(np.deg2rad(rotation_angle))
        self.n_traces = n_traces
        self.at_constrain_only = at_constrain_only
        self.results = None
        self.coeff = coeff
        self.w = w
        if coeff == 4:
            self.fourier_series = fourier_series
        if coeff == 7:
            self.fourier_series = fourier_series_2
        self.knowledge_constraints = GeologicalKnowledgeConstraints(self.x, self.constraints, self.coeff)

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
