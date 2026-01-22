from __future__ import annotations

from quantlab.metrics import annualized_sharpe

def main() -> None:
    # placeholder until we add data + backtest engine
    returns = [0.01, -0.01, 0.02, -0.02]
    print("Sharpe:", annualized_sharpe(returns, periods_per_year=12))

if __name__ == "__main__":
    main()
