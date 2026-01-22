from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from quantlab.backtest import long_short_momentum
from quantlab.config import load_yaml
from quantlab.metrics import annualized_sharpe
from quantlab.paths import make_run_dir
from quantlab.simdata import prices_to_returns, simulate_prices


def max_drawdown(equity: np.ndarray) -> float:
    peak = np.maximum.accumulate(equity)
    dd = equity / peak - 1.0
    return float(dd.min())


def main() -> None:
    cfg = load_yaml(Path(__file__).with_name("config.yaml"))

    run_dir = make_run_dir("flagship_equity_signals")
    (run_dir / "artifacts").mkdir(exist_ok=True)

    # Synthetic data for now (later: real data loader)
    prices = simulate_prices(
        n_assets=int(cfg.get("n_assets", 50)),
        n_days=int(cfg.get("n_days", 252 * 5)),
        seed=int(cfg.get("seed", 7)),
    )
    asset_rets = prices_to_returns(prices)

    port_rets = long_short_momentum(
        asset_returns=asset_rets,
        lookback=int(cfg.get("lookback", 252)),
        long_frac=float(cfg.get("long_frac", 0.2)),
        short_frac=float(cfg.get("short_frac", 0.2)),
        rebalance=str(cfg.get("rebalance", "monthly")),
        cost_bps=float(cfg.get("cost_bps", 5.0)),
    )

    # Metrics
    sharpe = annualized_sharpe(port_rets, periods_per_year=252)
    equity = np.cumprod(1.0 + port_rets)
    mdd = max_drawdown(equity)

    metrics = {
        "sharpe": float(sharpe),
        "max_drawdown": float(mdd),
        "n_days": int(asset_rets.shape[0]),
        "n_assets": int(asset_rets.shape[1]),
        "note": "Synthetic data run (placeholder until real data ingestion is added).",
    }

    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

    # Plot
    plt.figure()
    plt.plot(equity)
    plt.title("Equity Curve (Synthetic)")
    plt.xlabel("Day")
    plt.ylabel("Equity")
    plt.tight_layout()
    plt.savefig(run_dir / "equity_curve.png", dpi=150)

    print(f"Saved run to: {run_dir}")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
