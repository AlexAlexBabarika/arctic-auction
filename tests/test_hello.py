import numpy as np

from arctic.hello import EXPECTED, eisenberg_gale_prices, proportional_response_prices


def test_jax_proportional_response():
    prices = proportional_response_prices()
    np.testing.assert_allclose(prices, EXPECTED, atol=1e-3)


def test_clarabel_eisenberg_gale():
    prices = eisenberg_gale_prices("CLARABEL")
    np.testing.assert_allclose(prices, EXPECTED, atol=1e-3)
