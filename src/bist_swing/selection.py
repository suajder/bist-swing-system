from __future__ import annotations

from dataclasses import replace
from itertools import product
from typing import Dict, Iterable, List, Tuple

from .backtest import BacktestParams
from .signals import SignalParams


def expand_grid(
    grid: Dict[str, list],
    base_sp: SignalParams,
    base_bp: BacktestParams,
) -> List[Tuple[SignalParams, BacktestParams]]:
    """
    grid keys can target SignalParams or BacktestParams fields.
    """
    keys = list(grid.keys())
    vals = [grid[k] for k in keys]

    combos = []
    for v in product(*vals):
        sp = base_sp
        bp = base_bp
        for k, x in zip(keys, v):
            if hasattr(sp, k):
                sp = replace(sp, **{k: x})
            elif hasattr(bp, k):
                bp = replace(bp, **{k: x})
            else:
                raise ValueError(f"Unknown grid param: {k}")
        combos.append((sp, bp))
    return combos
