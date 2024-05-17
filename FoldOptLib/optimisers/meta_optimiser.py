from ..datatypes import SolverType
from abc import ABCMeta

class MetaOptimiser(ABCMeta):
    def __new__(mcls, name, bases, namespace):
        if 'optimise' not in namespace:
            namespace['optimise'] = mcls.optimise
        return super().__new__(mcls, name, bases, namespace)
    
    @staticmethod
    def optimise(self):
        """
        Runs the optimisation.

        Returns
        -------
        opt : Dict
            The result of the optimisation.

        Notes
        -----
        This function runs the optimisation by setting up the optimisation problem,
        checking if geological knowledge exists, and running the solver.
        """

        self.setup_optimisation()

        if (
            self._solver
            is self.optimiser._solvers[SolverType.DIFFERENTIAL_EVOLUTION]
        ):
            return self._solver(
                self.objective_function, self._bounds, init=self._guess
            )

        elif (
            self._solver
            is self.optimiser._solvers[SolverType.CONSTRAINED_TRUST_REGION]
        ):
            return self._solver(self.objective_function, x0=self._guess)

        # TODO: ...add support for restricted optimisation mode...

    attrs["optimise"] = optimise
    super().__init__(name, bases, attrs)
