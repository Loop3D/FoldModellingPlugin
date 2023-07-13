from typing import Union

from gaussian import gaussian_log_likelihood
import numpy as np
from scipy.stats import vonmises








class MaximumLikelihoodEstimation:
    """
    This is a base class for Maximum Likelihood Estimation for Fourier series or the axial surface.
    """

    def __init__(self, opt_type='fourier'):
        # check which objective function to use
        self.opt_type = opt_type
        if self.opt_type == 'fourier':
            self.mle = mle_fourier_series
        if self.opt_type == 'axial_surface':
            self.mle = mle_axial_surface

    def setup_mle_optimisation(self):
        if self.opt_type == 'fourier':
            self.mle = mle_fourier_series
        if self.opt_type == 'axial_surface':
            self.mle = mle_axial_surface

        return self.mle

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

    def __call__(self, *args, **kwargs):

        return self.mle
