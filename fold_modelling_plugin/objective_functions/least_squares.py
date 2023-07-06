import numpy as np


class LeastSquaresFunctions:
    def __init__(self):
        pass

    def square_residuals(self, theta):
        # objective function that calculates the square residuals Si
        # between observations y and optimised fourier series
        residuals = (np.tan(np.deg2rad(self.rotation_angle)) - np.tan(np.deg2rad(self.fourier_series(
                self.fold_frame, *theta)))) ** 2

        return residuals

    def huber_loss(self, residuals, delta=0.5):

        s = np.zeros(len(residuals))

        for i, residual in enumerate(residuals):
            if abs(residual) <= delta:
                s[i] = 0.5 * residual ** 2
            else:
                s[i] = delta * abs(residual) - 0.5 * delta ** 2

        return s

    def soft_l1_loss(self, residuals, delta=0.5):

        s = np.zeros(len(residuals))

        for i, residual in enumerate(residuals):
            s[i] = 2 * delta ** 2 * (np.sqrt(1 + (residual / delta) ** 2) - 1)

        return s

    def regularised_cost(self, theta):

        regularisation = 1e-50 * np.sum(np.abs(theta)) + 1e-50 * np.sum(theta ** 2)
        return 0.5 * np.sum(
            self.huber_loss(self.square_residuals(theta))) + regularisation + self.knowledge_constraints(theta)
