from scipy.optimize import minimize
# import knowledge_constraints
# importlib.reload(knowledge_constraints)
# from modified_loopstructural.modified_foldframe import FoldFrame
# from modified_loopstructural.extra_utils import *
from knowledge_constraints._helper import *
from knowledge_constraints.knowledge_constraints import GeologicalKnowledgeConstraints
from knowledge_constraints.splot_processor import SPlotProcessor
from knowledge_constraints.fourier_optimiser import FourierSeriesOptimiser
from LoopStructural import GeologicalModel
from LoopStructural.modelling.features.fold import FoldEvent
from LoopStructural.modelling.features.fold import FoldRotationAngle, SVariogram
from LoopStructural.modelling.features.fold import fourier_series
from LoopStructural.utils.helper import *
from geological_sampler.sampling_methods import *
from uncertainty_quantification.fold_uncertainty import *
import numpy as np
import pandas as pd
import mplstereonet
from sklearn.preprocessing import StandardScaler
from scipy.optimize import minimize, differential_evolution
from scipy.stats import vonmises_fisher


def huber_loss(residuals, delta=0.5):
    s = np.zeros(len(residuals))

    for i, residual in enumerate(residuals):
        if abs(residual) <= delta:
            s[i] = 0.5 * residual ** 2
        else:
            s[i] = delta * abs(residual) - 0.5 * delta ** 2

    return s


def calculate_semivariogram(fold_frame, fold_rotation, lag=None, nlag=60):
    svario = SVariogram(fold_frame, fold_rotation)
    svario.calc_semivariogram(lag=lag, nlag=nlag)
    wv = svario.find_wavelengths()
    theta = np.ones(4)
    theta[3] = wv[0]
    theta[0] = 0
    # py = wv[2]

    return theta, svario.lags, svario.variogram


def make_dataset_3(vec, xyz, name=None):
    # dgx = np.tile(vec, (len(xyz), 1))

    s2_dict = create_gradient_dict(x=xyz[:, 0],
                                   y=xyz[:, 1],
                                   z=xyz[:, 2],
                                   nx=vec[:, 0],
                                   ny=vec[:, 1],
                                   nz=vec[:, 2],
                                   feature_name=name, coord=0,
                                   data_type='foliation')
    dataset = pd.DataFrame()
    dataset = dataset.append(pd.DataFrame
                             (s2_dict, columns=['X', 'Y', 'Z', 'gx', 'gy', 'gz', 'feature_name', 'coord']))
    return dataset


def make_dataset_2(xyz, vec, coord=0):
    # vec = np.array(vec)
    # xyz = self.points

    dgx = np.tile(vec, (len(xyz), 1))

    s1_dict = create_gradient_dict(x=xyz[:, 0],
                                   y=xyz[:, 1],
                                   z=xyz[:, 2],
                                   nx=dgx[:, 0],
                                   ny=dgx[:, 1],
                                   nz=dgx[:, 2],
                                   feature_name='s1', coord=coord,
                                   data_type='foliation')
    # s1_dict = make_constant(s1_dict)
    dataset = pd.DataFrame()
    dataset = dataset.append(pd.DataFrame
                             (s1_dict, columns=['X', 'Y', 'Z', 'gx', 'gy', 'gz', 'feature_name', 'coord']))
    return dataset


def calculate_intersection_lineation(axial_surface, folded_foliation):
    """
    Calculate the intersection lineation of the axial surface and the folded foliation.

    Parameters:
    axial_surface (np.ndarray): The normal vector of the axial surface.
    folded_foliation (np.ndarray): The normal vector of the folded foliation.

    Returns:
    np.ndarray: The normalized intersection lineation vector.
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


def get_fold_curves(geological_feature, fold_axis=None):
    if fold_axis is not None:
        x = np.linspace(geological_feature.fold.foldframe[1].min(),
                        geological_feature.fold.foldframe[1].max(), 200)
        curve = geological_feature.fold.fold_axis_rotation(x)
        return x, curve

    if fold_axis is None:
        x = np.linspace(geological_feature.fold.foldframe[0].min(),
                        geological_feature.fold.foldframe[0].max(), 200)
        curve = geological_feature.fold.fold_limb_rotation(x)
        return x, curve





def scale(data):
    mms = StandardScaler()
    mms.fit(data)
    data_transformed = mms.transform(data)

    return data_transformed


# def logp(value: TensorVariable, mu: TensorVariable) -> TensorVariable:
#     return -(value - mu)**2


class AxialSurfaceOptimizer:
    def __init__(self, grid, bounding_box, folded_foliation_data, constraints, model,
                 axial_strike_dip=None,
                 wavelength_guess=None,
                 guess_method='plane_fit'):
        self.orientation_data = folded_foliation_data
        self.axial_strike_dip = axial_strike_dip
        self.initial_axial_guess = None
        self.wavelength_guess = wavelength_guess
        self.model = model  # self.model_initialisation()
        self.guess_method = guess_method
        self.grid = grid
        self.axial_surface = None
        self.optimised_axial_surface = None
        self.constraints = constraints
        self.points = self.model.rescale(self.model.data[['X', 'Y', 'Z']].to_numpy())
        self.fitted_params = None
        self.coords = self.model.data[['X', 'Y', 'Z']].to_numpy()
        self.fold = None
        self.bounding_box = bounding_box


    def initial_guess(self):

        if self.axial_strike_dip is not None:

            self.initial_axial_guess = self.axial_strike_dip

    def setup_optimisation():
        pass



    def find_axial_surface(self):
        init = self.initial_guess()
        x0 = self.initial_axial_guess
        # n = 30
        # V = generate_directional_training_dataset(x0,
        #                                           concentration=100,
        #                                           size=n)
        # init_pos = V['normals']
        # init_pos /= np.linalg.norm(init_pos, axis=1)[:, None]
        # del V
        # gc.collect()
        # min_nx = init_pos[:, 0].min()
        # max_nx = init_pos[:, 0].max()
        # min_ny = init_pos[:, 1].min()
        # max_ny = init_pos[:, 1].max()
        # min_nz = init_pos[:, 2].min()
        # max_nz = init_pos[:, 2].max()
        # bounds = np.array([(min_nx, max_nx),
        #                    (min_ny, max_ny),
        #                    (min_nz, max_nz)])
        # mu, sigma = 80, 10 # mean and standard deviation
        np.random.seed(180)
        dip = np.random.normal(80, 10, 20)
        strike = np.random.normal(110, 10, 20)
        mask = dip < 90
        dip = dip[mask]
        strike = strike[mask]
        init_pos = np.array([strike, dip]).T
        bounds = np.array([(0, 360), (0, 90)])  # np.array([(-1, 1), (-1, 1), (-1, 1)])
        best_pos = differential_evolution(self.mle_objective,
                                          bounds,
                                          # x0=[0.9999999, -0., 0.],
                                          init=init_pos,
                                          # init='sobol',
                                          strategy='best2exp',
                                          # mutation=(0.2, 0.45),
                                          polish=True,
                                          )
        x = best_pos.x
        x /= np.linalg.norm(x)

        self.optimised_axial_surface = x

        return self.optimised_axial_surface
