def safe_float(x, default=0.0):
    try:
        return float(x)
    except Exception:
        return default