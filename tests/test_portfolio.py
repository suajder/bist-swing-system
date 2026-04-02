import pytest
import pandas as pd
import numpy as np

from bist_swing.portfolio import r_summary

def test_r_summary_empty():
    df = pd.DataFrame()
    assert r_summary(df) == {}

def test_r_summary_basic():
    df = pd.DataFrame({
        "Type": ["ENTRY", "TP1", "STOP", "TP2"],
        "R_PnL": [np.nan, 2.0, -1.0, 3.0]
    })
    
    res = r_summary(df)
    assert res["n_exits"] == 3
    assert res["total_R"] == 4.0
    assert res["avg_R"] == pytest.approx(4.0 / 3)
    assert res["win_rate"] == pytest.approx(2.0 / 3)
