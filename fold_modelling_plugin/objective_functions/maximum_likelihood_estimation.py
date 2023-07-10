from typing import Union

from gaussian import gaussian_log_likelihood
import numpy as np
from scipy.stats import vonmises


class MaximumLikelihoodEstimation:
    """
    This is a base class for Maximum Likelihood Estimation for Fourier series or the axial surface.
    """

    def __init__(self, rotation_angle, fold_frame,
                 knowledge_constraints, folded_foliation_data, opt_type='fourier'):

        self.y = rotation_angle
        self.fold_frame = fold_frame
        self.knowledge_constraints = knowledge_constraints
        self.orientation_data = folded_foliation_data
        self.opt_type = opt_type
        self.mle = None

    def setup_optimisation(self):
        if self.opt_type == 'fourier':
            self.mle = self.mle_fourier_series
        if self.opt_type == 'axial_surface':
            self.mle = self.mle_axial_surface

        return self.mle

    def predicted_rotation_angle(self, theta):

        y_pred = np.tan(np.deg2rad(fourier_series(
            self.fold_frame, *theta)))

        return y_pred

    def loglikelihood(self, theta):

        y_pred = self.predicted_rotation_angle(theta)
        sigma = 1e-2
        likelihood = gaussian_log_likelihood(y, y_pred, sigma)
        return likelihood

    def mle_fourier_series(self, theta):

        log_likelihood = 0
        for fr, fd in zip(self.rotation_angle, self.fold_frame):
            log_likelihood += -self.loglikelihood(fr, fd, theta)

        if self.knowledge_constraints is None:
            total = log_likelihood
        else:
            total = (1e-2 * log_likelihood) + self.knowledge_constraints(theta)

        return total

    def mle_axial_surface(self, x: float) -> Union[int, float]:
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
        # mu = 0 because we want to minimises the angle between the observed and predicted folded foliation
        # kappa = 100 because we want to have a sharp distribution very close to the mean 0 (mu)
        mu = 0
        kappa = 100

        # Create a VonMises distribution with the given parameters
        vm = vonmises(mu, kappa)

        # Calculate the logpdf of the input array
        vm_logpdf = vm.logpdf(x)

        return vm_logpdf

    def mle_axial_surface(self, strike_dip):  # axial_normal):

        axial_normal = m_strike_dip_vector(strike_dip[0], strike_dip[1])[0]
        # init = self.initial_guess()
        axial_normal /= np.linalg.norm(axial_normal)
        self.build_fold_frame(axial_normal)
        flr, fld = self.calculate_fold_rotation_angle()
        # print('flr: ', flr)
        # print('fld: ', fld)
        # print('ith axial surface : ', axial_normal)
        opt = self.fit_fourier_series(fld, flr)
        self.fitted_params = opt.x
        # print('ith Fourier params :', self.fitted_params)

        predicted_bedding = self.get_predicted_bedding(fld, flr,
                                                       self.fitted_params)
        angle_difference = self.angle_difference(predicted_bedding,
                                                 self.orientation_data)
        del predicted_bedding, opt, flr, fld, self.axial_surface
        # angle_difference /= np.linalg.norm(angle_difference)
        # angle_difference = angle_difference.mean()
        # print('angle differences', angle_difference)

        # vm_logpdf = np.sum(-vM.logpdf(angle_difference, 10, loc=0))
        # mu = self.initial_axial_guess  # np.array([ 0.8 , 0.,  0.])
        # kappa = 5
        # vMF_logpdf = -vonmises_fisher_logp(axial_normal, mu, kappa)
        # log_likelihood = np.sum(-vonmises.logpdf(angle_difference, 0, 0.1))

        objective = angle_difference.sum()  # vm_logpdf
        del angle_difference
        gc.collect()
        # print('Objective fun:', objective)
        print(
            f"Axial surface optimisation...  \nAxial surface: {strike_dip} \nFourier params : {self.fitted_params} \nObjective fun : {objective}",
            end='\r', flush=True)
        # print('Axial surface: ', strike_dip, end='\r', flush=True)
        # print('Fourier params :', self.fitted_params, end='\r', flush=True)
        # print('Objective fun : ', objective, end='\r', flush=True)

        return objective

    def mle_objective_2(self, axial_normal):
        # init = self.initial_guess()
        axial_normal /= np.linalg.norm(axial_normal)
        # self.axial_surface = axial_normal
        initial_guess = self.initial_guess()
        fold_frame = self.build_fold_frame(axial_normal)
        fold = self.calculate_rotation_angles()
        print('ith axial surface : ', axial_normal)
        predicted_bedding = self.get_predicted_bedding_2(fold)
        angle_difference = self.angle_difference(predicted_bedding,
                                                 self.orientation_data)
        print('angle differences', angle_difference)

        # vm_logpdf = np.sum(-vM.logpdf(angle_difference, 10, loc=0))
        # mu = self.initial_axial_guess #np.array([ 0.8 , 0.,  0.])
        # kappa = 10
        # vMF_logpdf = -vonmises_fisher_logp(axial_normal, mu, kappa)
        objective = angle_difference.sum()

        print('Objective fun:', objective)

        return objective

    def optimisation(self):

        pass

    def __call__(self, *args, **kwargs):

        return self.optimisation()
