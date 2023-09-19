from typing import Callable, Dict, Any, Tuple

import numpy as np
from scipy.optimize import minimize, differential_evolution
import functools
from abc import ABC, abstractmethod

# TODO: merge this class with FoldOptimiser
class BaseOptimiser(ABC):
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

    @abstractmethod
    def optimise_with_trust_region(self, objective_function: Callable,
                                   x0: np.ndarray, constraints=None) -> Dict:
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

        opt = minimize(objective_function, x0,
                       method='trust-constr', jac='2-point',
                       constraints=constraints, **self.kwargs)

        return opt

    @abstractmethod
    def optimise_with_differential_evolution(self, objective_function: Callable, bounds: Tuple, init: str = 'halton',
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
                :param objective_function:
        """

        opt = differential_evolution(objective_function, bounds=bounds, init=init,
                                     maxiter=maxiter, seed=seed, polish=polish,
                                     strategy=strategy, mutation=mutation, **self.kwargs)

        return opt

    @abstractmethod
    def optimise(self, *args, **kwargs) -> Any:
        pass
