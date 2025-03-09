import numpy as np
import pytest
from ecopystats.diversity import diversity

def test_shannon_basic():
    arr = np.array([[10, 10, 10]])  # 3 equally abundant species
    # Shannon (raw) => -sum(1/3 * ln(1/3)) = ln(3) ~ 1.0986
    result = diversity(arr, method='shannon', axis=1, base=np.e, numbers_equivalent=False)
    assert pytest.approx(result[0], 0.001) == 1.0986

def test_shannon_num_equiv():
    arr = np.array([[10, 10, 10]])
    # Shannon => ln(3). num_equiv => exp(ln(3)) = 3
    result = diversity(arr, method='shannon', axis=1, numbers_equivalent=True)
    assert pytest.approx(result[0], 0.001) == 3.0

def test_simpson_basic():
    arr = np.array([[10, 10, 10]])
    # Simpson => sum( (1/3)^2 ) = 1/3 ~ 0.3333
    result = diversity(arr, method='simpson', axis=1)
    assert pytest.approx(result[0], 0.001) == 0.3333

