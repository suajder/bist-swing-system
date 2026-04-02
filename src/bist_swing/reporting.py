from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_equity(
    equity: pd.DataFrame,
    outpath: str | Path,
    title: str = "Equity Curve",
) -> Path:
    """
    Save a simple equity curve plot.

    Parameters
    ----------
    equity:
        DataFrame with a datetime-like index and at least one numeric column.
        Typical input: eqdf[["Equity"]]
    outpath:
        Destination filepath (PNG recommended).
    title:
        Plot title.

    Returns
    -------
    Path to the saved file.
    """
    outpath = Path(outpath)
    outpath.parent.mkdir(parents=True, exist_ok=True)

    if equity is None or len(equity) == 0:
        raise ValueError("equity is empty")

    # Ensure we plot numeric cols only
    cols = [c for c in equity.columns if pd.api.types.is_numeric_dtype(equity[c])]
    if not cols:
        raise ValueError("equity has no numeric columns to plot")

    y = equity[cols[0]].copy()

    plt.figure()
    plt.plot(y.index, y.values)
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel(cols[0])
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()

    return outpath
