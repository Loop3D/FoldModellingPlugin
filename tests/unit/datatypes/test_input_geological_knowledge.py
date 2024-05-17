import pytest
from FoldOptLib import InputGeologicalKnowledge, KnowledgeType
from FoldOptLib.datatypes.probability_distributions import (
    NormalDistribution,
    VonMisesFisherDistribution,
)
from FoldOptLib.utils import strike_dip_to_vector


def test_post_init():
    # _axial_surface = NormalDistribution(mu=[1, 2], sigma=1)
    # _knowledge = InputGeologicalKnowledge(axial_surface=_axial_surface)
    axial_surface = VonMisesFisherDistribution(mu=[1, 2], kappa=1)
    knowledge = InputGeologicalKnowledge(axial_surface=axial_surface)
    assert len(knowledge.axial_surface.mu) == 3


def test_getitem():
    asymmetry = NormalDistribution(mu=1, sigma=1)
    axial_trace = NormalDistribution(mu=2, sigma=2)
    knowledge = InputGeologicalKnowledge(asymmetry=asymmetry, axial_trace=axial_trace)

    assert knowledge[KnowledgeType.ASYMMETRY] == asymmetry
    assert knowledge[KnowledgeType.AXIAL_TRACE] == axial_trace
    assert knowledge[KnowledgeType.WAVELENGTH] is None
