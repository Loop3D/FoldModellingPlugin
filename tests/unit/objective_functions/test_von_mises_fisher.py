import pytest
import numpy as np
from FoldOptLib.objective_functions import VonMisesFisher


@pytest.fixture
def vmf():
    mu = [1, 0, 0]
    kappa = 1
    return VonMisesFisher(mu, kappa)


def test_init_valid_input(vmf):
    assert vmf is not None


def test_init_invalid_mu():
    with pytest.raises(ValueError):
        VonMisesFisher([1, 0], 1)


def test_init_invalid_kappa():
    with pytest.raises(ValueError):
        VonMisesFisher([1, 0, 0], -1)


def test_pdf(vmf):
    x = np.array([[1, 0, 0], [0, 1, 0]])
    result = vmf.pdf(x)
    assert result.shape == (2,)


def test_logpdf(vmf):
    x = np.array([[1, 0, 0], [0, 1, 0]])
    result = vmf.logpdf(x)
    assert result.shape == (2,)


def test_draw_samples_valid_input(vmf):
    samples = vmf.draw_samples(size=5, random_state=42)
    assert samples.shape == (5, 3)


def test_draw_samples_invalid_size(vmf):
    with pytest.raises(TypeError):
        vmf.draw_samples(size="5", random_state=42)


def test_draw_samples_invalid_random_state(vmf):
    with pytest.raises(TypeError):
        vmf.draw_samples(size=5, random_state="42")
