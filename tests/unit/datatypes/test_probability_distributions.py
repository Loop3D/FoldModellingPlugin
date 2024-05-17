import pytest
import numpy as np
from FoldOptLib.datatypes.probability_distributions import (
    Bounds,
    NormalDistribution,
    VonMisesFisherDistribution,
)


def test_bounds():
    bounds = Bounds(lower_bound=0, upper_bound=10)
    assert bounds.lower_bound == 0
    assert bounds.upper_bound == 10


def test_normal_distribution():
    normal_dist = NormalDistribution(mu=0, sigma=1, weight=0.5)
    assert normal_dist.mu == 0
    assert normal_dist.sigma == 1
    assert normal_dist.weight == 0.5


def test_von_mises_fisher_distribution():
    von_mises_fisher_dist = VonMisesFisherDistribution(
        mu=[0, 0, 0], kappa=1, weight=0.5
    )
    assert von_mises_fisher_dist.mu == [0, 0, 0]
    assert von_mises_fisher_dist.kappa == 1
    assert von_mises_fisher_dist.weight == 0.5


def test_bounds_with_numpy():
    bounds = Bounds(lower_bound=np.array([0, 1]), upper_bound=np.array([10, 11]))
    assert np.array_equal(bounds.lower_bound, np.array([0, 1]))
    assert np.array_equal(bounds.upper_bound, np.array([10, 11]))


def test_normal_distribution_with_numpy():
    normal_dist = NormalDistribution(mu=1, sigma=1, weight=1)
    assert np.array_equal(normal_dist.mu, 1)
    assert np.array_equal(normal_dist.sigma, 1)
    assert np.array_equal(normal_dist.weight, 1)


def test_von_mises_fisher_distribution_with_numpy():
    von_mises_fisher_dist = VonMisesFisherDistribution(
        mu=np.array([0, 10, 1]), kappa=1, weight=1
    )
    assert np.array_equal(von_mises_fisher_dist.mu, np.array([0, 10, 1]))
    assert np.array_equal(von_mises_fisher_dist.kappa, 1)
    assert np.array_equal(von_mises_fisher_dist.weight, 1)
