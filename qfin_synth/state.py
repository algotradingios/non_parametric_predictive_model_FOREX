import numpy as np
import pandas as pd

def compute_basic_state(
    df: pd.DataFrame,
    price_col: str = "price",
    time_col: str | None = "time",
    vol_window: int = 60,
    alpha: float | None = None,
    warmup: int | None = None,
    eps: float = 1e-12,
):
    """
    Constructs Option-1 state and supervised pairs:

      r_t = log(P_t / P_{t-1})
      sigma2_t = (1-alpha)*sigma2_{t-1} + alpha*r_t^2   (EWMA)
      sigma2_0 = Var(r_1..r_warmup) (ddof=1)
      lvol_t = log(sigma_hat_t) = 0.5*log(sigma2_t)

    Training pairs:
      X = (r_{t-1}, lvol_{t-1})
      y = r_t

    Returns:
      df_out with columns r, sigma_hat, lvol, r_lag1, lvol_lag1
      X: np.ndarray shape [n,2]
      y: np.ndarray shape [n]
      meta: dict (alpha, warmup, init_sigma_recommended, etc.)
    """
    df = df.copy()
    if time_col and time_col in df.columns:
        df = df.sort_values(time_col).reset_index(drop=True)

    if price_col not in df.columns:
        raise ValueError(f"Missing price column '{price_col}'")

    price = df[price_col].astype(float).to_numpy()
    if np.any(price <= 0):
        raise ValueError("All prices must be > 0 for log-returns.")

    if alpha is None:
        alpha = 2.0 / (vol_window + 1.0)
    if not (0.0 < alpha < 1.0):
        raise ValueError("alpha must be in (0,1).")

    if warmup is None:
        warmup = vol_window
    warmup = int(warmup)
    if warmup < 5:
        raise ValueError("warmup too small; use at least 5 (preferably vol_window).")

    logp = np.log(price)
    r = np.full_like(logp, np.nan, dtype=float)
    r[1:] = np.diff(logp)

    valid_r = r[1:]
    if len(valid_r) < warmup + 2:
        raise ValueError("Not enough data for warmup variance and lagged states.")

    # Initial variance from first warmup returns
    init_slice = valid_r[:warmup]
    sigma2_0 = float(np.var(init_slice, ddof=1))
    sigma2_0 = max(sigma2_0, eps)

    sigma2 = np.full_like(r, np.nan, dtype=float)
    sigma2[warmup] = sigma2_0
    for t in range(warmup + 1, len(r)):
        sigma2[t] = (1.0 - alpha) * sigma2[t - 1] + alpha * (r[t] ** 2)

    sigma2 = np.maximum(sigma2, eps)
    sigma_hat = np.sqrt(sigma2)
    lvol = 0.5 * np.log(sigma2)  # log(sigma_hat)

    df["logp"] = logp
    df["r"] = r
    df["sigma_hat"] = sigma_hat
    df["lvol"] = lvol
    df["r_lag1"] = df["r"].shift(1)
    df["lvol_lag1"] = df["lvol"].shift(1)

    df_out = df.dropna(subset=["r", "r_lag1", "lvol_lag1"]).reset_index(drop=True)

    X = df_out[["r_lag1", "lvol_lag1"]].to_numpy(dtype=float)
    y = df_out["r"].to_numpy(dtype=float)

    init_sigma_rec = float(np.nanmedian(df_out["sigma_hat"].to_numpy()))
    init_sigma_rec = max(init_sigma_rec, 1e-6)

    meta = {
        "vol_window": int(vol_window),
        "alpha": float(alpha),
        "warmup": int(warmup),
        "state_cols": ["r_lag1", "lvol_lag1"],
        "target_col": "r",
        "init_sigma_recommended": init_sigma_rec,
    }
    
    return df_out, X, y, meta
