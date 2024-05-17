from typing import Callable, Dict, Any, Tuple, Union, List
import numpy as np
from ..utils.utils import *
from scipy.optimize import minimize, differential_evolution
from ..datatypes import SolverType
import beartype


@beartype.beartype
class Solver:
    """
    Base class for fold geometry optimisation.
    """

    def __init__(self, solver="differential_evolution", **kwargs: Dict[str, Any]):
        """
        Constructs all the necessary attributes for the Fold Optimiser object.

        Parameters
        ----------
            kwargs : dict
                Additional keyword arguments.
        """
        self.solver = solver
        self.kwargs = kwargs

    @beartype.beartype
    @staticmethod
    def differential_evolution(
        objective_function: Callable,
        bounds: Union[Tuple, List],
        init: str = "halton",
        maxiter: int = 5000,
        seed: int = 80,
        polish: bool = True,
        strategy: str = "best2exp",
        mutation: Tuple[float, float] = (0.3, 0.99),
        **kwargs,
    ) -> Dict:
        """
        Solves the optimization problem using the differential evolution method.
        Check Scipy documentation for more info

        Parameters
        ----------
            objective_function: Callable
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
                The mutation constant. Default is (0.3, 0.99) and it was tested and have proven to explore the parameter
                space efficiently.

        Returns
        -------
            opt : Dict
                The solution of the optimization.

        """

        opt = differential_evolution(
            objective_function,
            bounds=bounds,
            init=init,
            maxiter=maxiter,
            seed=seed,
            polish=polish,
            strategy=strategy,
            mutation=mutation,
            **kwargs,
        )

        return opt

    @beartype.beartype
    @staticmethod
    def constrained_trust_region(
        objective_function: Callable, x0: np.ndarray, constraints=None, **kwargs
    ) -> Dict:
        """
        Solves the optimisation problem using the trust region method.

        Parameters
        ----------

            objective_function: Callable
                The objective function to be optimised.
            x0 : np.ndarray
                Initial guess of the parameters to be optimised.

        Returns
        -------
            opt : Dict
                The solution of the optimisation.

        """

        opt = minimize(
            objective_function,
            x0,
            method="trust-constr",
            jac="2-point",
            constraints=constraints,
            **kwargs,
        )

        return opt

    @beartype.beartype
    def __getitem__(self, solver_type: SolverType):
        """
        Calls the solver function based on the solver type.

        Parameters
        ----------
            solver_type : SolverType
                The type of solver to use.

        Returns
        -------
            solver : Callable
                The solver function.
        """
        # Map the solver type to the solver function
        solver_map = {
            SolverType.DIFFERENTIAL_EVOLUTION: self.differential_evolution,
            SolverType.CONSTRAINED_TRUST_REGION: self.constrained_trust_region,
            SolverType.UNCONSTRAINED_TRUST_REGION: None,
            SolverType.PARTICLE_SWARM: None,
        }
        return solver_map[solver_type]
