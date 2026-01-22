from __future__ import annotations

import numpy as np


def simulate_prices(
    n_assets: int = 50,
    n_days: int = 252 * 5,
    seed: int = 7,
) -> np.ndarray:
    """
    Simple geometric random walk (synthetic prices).
    Returns: prices shape (n_days, n_assets)
    """
    rng = np.random.default_rng(seed)
    # daily returns ~ N(mu, sigma)
    mu = 0.0002
    sigma = 0.01
    rets = rng.normal(loc=mu, scale=sigma, size=(n_days, n_assets))
    prices = 100.0 * np.exp(np.cumsum(rets, axis=0))
    return prices


def prices_to_returns(prices: np.ndarray) -> np.ndarray:
    # simple returns
    return prices[1:] / prices[:-1] - 1.0
