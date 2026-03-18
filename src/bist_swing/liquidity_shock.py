def add_liquidity_shock_cols(df):

    vol = df["Volume"]
    close = df["Close"]

    vol_ma20 = vol.rolling(20).mean()

    vol_spike = vol / vol_ma20

    range_pct = (df["High"] - df["Low"]) / close

    df["liq_shock"] = (vol_spike > 2.0) & (range_pct < 0.05)

    return df