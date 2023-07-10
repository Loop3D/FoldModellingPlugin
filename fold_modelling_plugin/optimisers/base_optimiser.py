from typing import Callable, Dict, Any, Tuple

import numpy as np
from scipy.optimize import minimize, differential_evolution


class BaseOptimiser:
    """
    A base class that to represent an abstract Optimiser.

    ...

    Attributes
    ----------
    objective_function : Callable
        a function which is to be minimised.
    kwargs : Dict[str, Any]
        additional keyword arguments.

    Methods
    -------
    solve_with_trust_region()
        Solves the optimisation problem using the trust region method.
    solve_with_differential_evolution()
        Solves the optimisation problem using the differential evolution method.
    """

    def __init__(self, objective_function: Callable, **kwargs: Dict[str, Any]) -> None:
        """
        Constructs all the necessary attributes for the optimiser object.

        Parameters
        ----------
            objective_function : Callable
                a function which is to be minimized.
            **kwargs : Dict[str, Any]
                additional keyword arguments.
        """
        self.objective_function = objective_function
        self.kwargs = kwargs

    def solve_with_trust_region(self, x0: np.ndarray) -> Dict:
        """
        Solves the optimization problem using the trust region method.

        Parameters
        ----------
            x0 : np.ndarray
                Initial guess of the parameters to be optimised.

        Returns
        -------
            opt : Dict
                The solution of the optimisation.
        """

        opt = minimize(self.objective_function, x0,
                       method='trust-constr', jac='2-point', **self.kwargs)

        return opt

    def solve_with_differential_evolution(self, bounds: Tuple, init: str = 'halton',
                                          maxiter: int = 5000, seed: int = 80,
                                          polish: bool = True, strategy: str = 'best2exp',
                                          mutation: Tuple[float, float] = (0.3, 0.99)) -> Dict:
        """
        Solves the optimization problem using the differential evolution method.
        Check Scipy documentation for more info

        Parameters
        ----------
            bounds : Tuple
                Bounds for variables. ``(min, max)`` pairs for each element in ``x``,
                defining the bounds on that parameter.
            init : str
                Specify how population initialisation is performed. Default is 'halton'.
            maxiter : int
                The maximum number of generations over which the entire population is evolved. Default is 5000.
            seed : int
                The seed for the pseudo-random number generator. Default is 80.
            polish : bool
                If True (default), then differential evolution is followed by a polishing phase.
            strategy : str
                The differential evolution strategy to use. Default is 'best2exp'.
            mutation : Tuple[float, float]
                The mutation constant. Default is (0.3, 0.99).

        Returns
        -------
            opt : Dict
                The solution of the optimization.
        """

        opt = differential_evolution(self.objective_function, bounds=bounds, init=init,
                                     maxiter=maxiter, seed=seed, polish=polish,
                                     strategy=strategy, mutation=mutation, **self.kwargs)

        return opt

    #TODO: Add a method to solve the optimisation problem
