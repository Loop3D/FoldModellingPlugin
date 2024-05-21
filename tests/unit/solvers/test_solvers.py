from FoldOptLib.solvers import Solver
from FoldOptLib.datatypes import SolverType
import numpy as np


def test_solver_init():
    solver = Solver(solver="differential_evolution")
    assert solver.solver == "differential_evolution"
    assert solver.kwargs == {}


def test_solver_differential_evolution():
    def objective_function(x):
        return x[0] ** 2 + x[1] ** 2

    bounds = [(-1, 1), (-1, 1)]
    solver = Solver()
    result = solver.differential_evolution(objective_function, bounds)
    assert result.success is True
    assert np.allclose(result.x, [0, 0], atol=1e-2)


def test_solver_constrained_trust_region():
    def objective_function(x):
        return x[0] ** 2 + x[1] ** 2

    x0 = np.array([1, 1])
    solver = Solver()
    result = solver.constrained_trust_region(objective_function, x0)
    assert result.success is True
    assert np.allclose(result.x, [0, 0], atol=1e-2)


def test_solver_getitem():
    solver = Solver()
    assert solver[SolverType.DIFFERENTIAL_EVOLUTION] == solver.differential_evolution
    assert (
        solver[SolverType.CONSTRAINED_TRUST_REGION] == solver.constrained_trust_region
    )
    assert solver[SolverType.UNCONSTRAINED_TRUST_REGION] == None
    assert solver[SolverType.PARTICLE_SWARM] == None
