from __future__ import annotations

import math
from typing import Iterable


def annualized_sharpe(returns: Iterable[float], periods_per_year: int = 252) -> float:
    r = list(returns)
    if len(r) < 2:
        return float("nan")
    mean = sum(r) / len(r)
    var = sum((x - mean) ** 2 for x in r) / (len(r) - 1)
    std = math.sqrt(var)
    if std == 0:
        return float("nan")
    return (mean / std) * math.sqrt(periods_per_year)
