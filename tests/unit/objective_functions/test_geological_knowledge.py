import pytest
import numpy

# Import the class to be tested
from FoldOptLib.objective_functions import GeologicalKnowledgeFunctions
from FoldOptLib.datatypes import InputGeologicalKnowledge
from FoldOptLib.datatypes.probability_distributions import NormalDistribution, VonMisesFisherDistribution


@pytest.fixture
def geological_knowledge():
    # Sample constraints and x values for testing
    # Example of InputGeologicalKnowledge filled
    knowledge = InputGeologicalKnowledge(
        asymmetry=NormalDistribution(mu=0, sigma=1),
        fold_wavelength=NormalDistribution(mu=10, sigma=2),
        axis_wavelength=NormalDistribution(mu=5, sigma=1),
        tightness=NormalDistribution(mu=0.5, sigma=0.1),
        hinge_line=NormalDistribution(mu=45, sigma=5),
        axial_trace=NormalDistribution(mu=30, sigma=3),
        axial_surface=VonMisesFisherDistribution(mu=[0.5, 0.5, 0.5], kappa=10),
    )

    gkf = GeologicalKnowledgeFunctions(knowledge)
    gkf.x = numpy.linspace(-1, 1, 100)

    return gkf


def test_axial_surface_objective_function(geological_knowledge):
    x = [0.0, 0.0, 1.0]
    result = geological_knowledge.axial_surface_objective_function(x)
    assert isinstance(result, float) 

def test_invalid_axial_surface_objective_function(geological_knowledge):
    x = [0.0, 0.0, 1.0, 1.0]
    with pytest.raises(ValueError):
        geological_knowledge.axial_surface_objective_function(x)

def test_axial_trace_objective_function(geological_knowledge):
    theta = numpy.array([0.0, 1.0, 1.0, 500.0])
    result = geological_knowledge.axial_trace_objective_function(theta)
    assert isinstance(result, (float)) or isinstance(result, (int)) or isinstance(result, (list, numpy.ndarray))


def test_wavelength_objective_function(geological_knowledge):
    theta = numpy.array([0, 1, 1, 500])
    result = geological_knowledge.wavelength_objective_function(theta)
    assert isinstance (result, float)


def test_fold_axis_wavelength_objective_function(geological_knowledge):
    theta = numpy.array([0, 1, 1, 500])
    result = geological_knowledge.fold_axis_wavelength_objective_function(theta)
    assert isinstance(result, float)


def test_tightness_objective_function(geological_knowledge):
    theta = numpy.array([0, 1, 1, 500])
    result = geological_knowledge.tightness_objective_function(theta)
    assert isinstance(result, float)

def test_hinge_angle_objective_function(geological_knowledge):
    theta = numpy.array([0, 1, 1, 500])
    result = geological_knowledge.hinge_angle_objective_function(theta)
    assert isinstance(result, float)

def test_asymmetry_objective_function(geological_knowledge):
    theta = numpy.array([0, 1, 1, 500])
    result = geological_knowledge.asymmetry_objective_function(theta)
    assert isinstance(result, float)

def test_call_fourier_series(geological_knowledge):
    theta = numpy.array([0, 1, 1, 500])
    result = geological_knowledge(theta)
    assert isinstance(result, float)

def test_call_valid_vector(geological_knowledge):
    vector = numpy.array([0, 1, 1], dtype=float)
    vector /= numpy.linalg.norm(vector)
    result_1 = geological_knowledge(vector)
    assert isinstance(result_1, float)
    
def test_call_invalid_vector(geological_knowledge):
    vector = numpy.array([0, 1, 1, 500, 0.5, 45, 30, 0.5, 0.5, 0.5])
    with pytest.raises(ValueError):
        geological_knowledge(vector)
    