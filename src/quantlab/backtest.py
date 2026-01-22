from __future__ import annotations

import numpy as np


def rebalance_step(rebalance: str) -> int:
    m = {
        "daily": 1,
        "weekly": 5,
        "monthly": 21,
    }
    if rebalance not in m:
        raise ValueError(f"Unknown rebalance '{rebalance}'. Use: {list(m)}")
    return m[rebalance]


def long_short_momentum(
    asset_returns: np.ndarray,  # shape (T, N)
    lookback: int = 252,
    long_frac: float = 0.2,
    short_frac: float = 0.2,
    rebalance: str = "monthly",
    cost_bps: float = 5.0,
) -> np.ndarray:
    """
    Cross-sectional momentum:
    - signal = past lookback cumulative return
    - long top quantile, short bottom quantile
    - equal-weight within long/short
    - costs applied on turnover at rebalance dates (bps of traded notional)
    Returns: portfolio returns array shape (T,)
    """
    T, N = asset_returns.shape
    step = rebalance_step(rebalance)
    port_rets = np.zeros(T)

    w_prev = np.zeros(N)

    for t in range(lookback, T):
        if (t - lookback) % step == 0:
            # signal: cumulative return over lookback window
            window = asset_returns[t - lookback : t]  # (lookback, N)
            sig = np.prod(1.0 + window, axis=0) - 1.0

            n_long = max(1, int(round(long_frac * N)))
            n_short = max(1, int(round(short_frac * N)))

            idx = np.argsort(sig)
            short_idx = idx[:n_short]
            long_idx = idx[-n_long:]

            w = np.zeros(N)
            w[long_idx] = 1.0 / n_long
            w[short_idx] = -1.0 / n_short

            # turnover + cost (bps)
            turnover = np.sum(np.abs(w - w_prev))
            cost = (cost_bps / 1e4) * turnover  # bps -> fraction
            w_prev = w
        else:
            cost = 0.0

        port_rets[t] = float(np.dot(w_prev, asset_returns[t]) - cost)

    return port_rets
