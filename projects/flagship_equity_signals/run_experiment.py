from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from quantlab.backtest import long_short_momentum
from quantlab.config import load_yaml
from quantlab.data import fetch_prices_yfinance
from quantlab.metrics import annualized_sharpe
from quantlab.paths import make_run_dir


def max_drawdown(equity: np.ndarray) -> float:
    peak = np.maximum.accumulate(equity)
    dd = equity / peak - 1.0
    return float(dd.min())


def main() -> None:
    cfg = load_yaml(Path(__file__).with_name("config.yaml"))
    run_dir = make_run_dir("flagship_equity_signals")

    data_cfg = cfg["data"]
    strat_cfg = cfg["strategy"]

    if data_cfg["source"] != "yfinance":
        raise ValueError("Only data.source=yfinance is implemented in this step.")

    prices = fetch_prices_yfinance(
        tickers=data_cfg["tickers"],
        start=data_cfg["start"],
        end=data_cfg["end"],
        refresh=bool(data_cfg.get("refresh", False)),
    )

    # returns matrix (T, N)
    rets_df = prices.pct_change().dropna(how="all")
    rets_df = rets_df.dropna(axis=1, how="any")  # keep only tickers with full data
    asset_rets = rets_df.to_numpy()

    port_rets = long_short_momentum(
        asset_returns=asset_rets,
        lookback=int(strat_cfg["lookback"]),
        long_frac=float(strat_cfg["long_frac"]),
        short_frac=float(strat_cfg["short_frac"]),
        rebalance=str(strat_cfg["rebalance"]),
        cost_bps=float(strat_cfg["cost_bps"]),
    )

    sharpe = annualized_sharpe(port_rets, periods_per_year=252)
    equity = np.cumprod(1.0 + port_rets)
    mdd = max_drawdown(equity)

    metrics = {
        "tickers_used": list(rets_df.columns),
        "n_days": int(asset_rets.shape[0]),
        "n_assets": int(asset_rets.shape[1]),
        "sharpe": float(sharpe),
        "max_drawdown": float(mdd),
        "note": "Prices via yfinance (auto_adjust=True). Universe is small ETF basket for reproducibility.",
    }

    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

    plt.figure()
    plt.plot(equity)
    plt.title("Equity Curve (Real Data)")
    plt.xlabel("Day")
    plt.ylabel("Equity")
    plt.tight_layout()
    plt.savefig(run_dir / "equity_curve.png", dpi=150)

    print(f"Saved run to: {run_dir}")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
