import pytest
import numpy as np
from FoldOptLib.splot.splot_processor import SPlotProcessor
from FoldOptLib.utils.utils import fourier_series_x_intercepts, fourier_series


def test_init():
    splot_processor = SPlotProcessor()
    assert splot_processor.x is None
    assert splot_processor.splot_function_map == {4: fourier_series}
    assert splot_processor.intercept_function_map == {4: fourier_series_x_intercepts}


def test_find_amax_amin():
    splot_processor = SPlotProcessor()
    x = np.linspace(0, 10, 100)
    splot_processor.x = x

    theta = np.array([0, 1, 1, 500])
    amax, amin = splot_processor.find_amax_amin(theta)
    assert amin < amax


def test_calculate_splot():
    splot_processor = SPlotProcessor()
    x = np.linspace(0, 10, 100)
    splot_processor.x = x

    theta = np.array([0, 1, 1, 500])
    curve = splot_processor.calculate_splot(x, theta)
    assert len(curve) == len(x)


def test_calculate_tightness():
    splot_processor = SPlotProcessor()
    x = np.linspace(0, 10, 100)
    splot_processor.x = x

    theta = np.array([0, 1, 1, 500])
    tightness = splot_processor.calculate_tightness(theta)
    assert tightness > 0.0
    assert tightness < 180.0


def test_calculate_asymmetry():
    splot_processor = SPlotProcessor()
    x = np.linspace(0, 10, 100)
    splot_processor.x = x

    theta = np.array([0, 1, 1, 500])
    asymmetry = splot_processor.calculate_asymmetry(theta)
    assert asymmetry >= 0.0
