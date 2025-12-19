# qfin_synth/validation.py
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

def _acf(x: np.ndarray, max_lag: int = 20) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    x = x - x.mean()
    denom = np.dot(x, x) + 1e-12
    out = []
    for k in range(1, max_lag + 1):
        out.append(float(np.dot(x[:-k], x[k:]) / denom))
    return np.array(out, dtype=float)

def _max_drawdown(prices: np.ndarray) -> float:
    p = np.asarray(prices, dtype=float)
    running_max = np.maximum.accumulate(p)
    dd = (p / (running_max + 1e-12)) - 1.0
    return float(dd.min())

def validate_simulator(
    real_prices: pd.Series,
    synth_prices_df: pd.DataFrame,
    price_col_synth: str = "price",
    path_col: str = "path",
    max_lag: int = 20,
    vol_window: int = 20,
) -> dict:
    """
    real_prices: pd.Series (historical close_price)
    synth_prices_df: DataFrame with columns [path, price] (plus t)
    Returns a dict with diagnostics and pass/fail suggestion.
    """
    # Real returns
    r_real = np.log(real_prices).diff().dropna().to_numpy()

    # Synthetic returns pooled across paths
    synth_returns = []
    synth_drawdowns = []
    for _, g in synth_prices_df.groupby(path_col):
        p = g[price_col_synth].astype(float).to_numpy()
        rr = np.diff(np.log(p))
        synth_returns.append(rr)
        synth_drawdowns.append(_max_drawdown(p))
    r_synth = np.concatenate(synth_returns) if len(synth_returns) else np.array([])

    # Basic moments
    stats = {
        "real_mean": float(np.mean(r_real)),
        "real_std": float(np.std(r_real, ddof=1)),
        "real_kurt": float(pd.Series(r_real).kurt()),
        "synth_mean": float(np.mean(r_synth)),
        "synth_std": float(np.std(r_synth, ddof=1)),
        "synth_kurt": float(pd.Series(r_synth).kurt()),
    }

    # KS tests on returns distribution and realized vol distribution
    ks_ret = ks_2samp(r_real, r_synth)
    stats["ks_ret_stat"] = float(ks_ret.statistic)
    stats["ks_ret_pvalue"] = float(ks_ret.pvalue)

    # Realized vol (rolling std of returns)
    rv_real = pd.Series(r_real).rolling(vol_window).std().dropna().to_numpy()
    rv_synth = pd.Series(r_synth).rolling(vol_window).std().dropna().to_numpy()
    ks_rv = ks_2samp(rv_real, rv_synth)
    stats["ks_rv_stat"] = float(ks_rv.statistic)
    stats["ks_rv_pvalue"] = float(ks_rv.pvalue)

    # ACF
    stats["acf_ret_real"] = _acf(r_real, max_lag=max_lag).tolist()
    stats["acf_ret_synth"] = _acf(r_synth, max_lag=max_lag).tolist()
    stats["acf_absret_real"] = _acf(np.abs(r_real), max_lag=max_lag).tolist()
    stats["acf_absret_synth"] = _acf(np.abs(r_synth), max_lag=max_lag).tolist()

    # Drawdowns
    stats["real_max_drawdown"] = _max_drawdown(real_prices.to_numpy())
    stats["synth_drawdown_mean"] = float(np.mean(synth_drawdowns))
    stats["synth_drawdown_p10"] = float(np.quantile(synth_drawdowns, 0.10))
    stats["synth_drawdown_p90"] = float(np.quantile(synth_drawdowns, 0.90))

    # Heurística simple de “apto/no apto” (ajusta umbrales a tu caso)
    # - Retornos: KS stat no demasiado grande (p.ej. < 0.15)
    # - Volatilidad: KS stat < 0.20
    # - ACF(ret) baja en lags cortos y ACF(|ret|) positiva (clustering)
    ok = True
    if stats["ks_ret_stat"] > 0.15:
        ok = False
    if stats["ks_rv_stat"] > 0.20:
        ok = False
    stats["simulator_ok"] = bool(ok)
    return stats
