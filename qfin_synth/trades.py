# qfin_synth/trades.py
from __future__ import annotations

import numpy as np
import pandas as pd
import ta
from tqdm import tqdm


def triple_barrier_labels(
    prices: pd.Series,
    entry_idx: int,
    H: int = 20,
    tp_mult: float = 1.5,
    sl_mult: float = 1.0,
    method: str = "atr",
    past_bars: int = 50,
    high_prices: pd.Series | None = None,
    low_prices: pd.Series | None = None,
) -> tuple[int, int, float]:
    """
    Returns: (label, exit_idx, exit_rt)
      label =  1  -> TP touched first
      label = -1  -> SL touched first
      label =  0  -> horizon reached first
    exit_rt is an ABSOLUTE price difference (prices[t] - p0).
    
    Args:
        prices: Close prices series
        entry_idx: Entry index
        H: Horizon (bars)
        tp_mult: Take profit multiplier
        sl_mult: Stop loss multiplier
        method: "atr" or "std"
        past_bars: Window for volatility estimation
        high_prices: Optional high prices series (for ATR calculation)
        low_prices: Optional low prices series (for ATR calculation)
    """
    end = min(entry_idx + H, len(prices) - 1)
    p0 = prices.iloc[entry_idx]

    # Local volatility estimation
    if entry_idx >= past_bars:
        window = past_bars
    else:
        window = max(entry_idx - 1, 2)
    
    # Use proper ATR calculation when high/low are available
    if method == "atr" and high_prices is not None and low_prices is not None:
        # Calculate ATR using ta library - need data up to entry_idx
        # ATR needs at least 'window' bars of data
        start_idx = max(0, entry_idx - window + 1)
        atr_indicator = ta.volatility.AverageTrueRange(
            high=high_prices.iloc[start_idx:entry_idx + 1],
            low=low_prices.iloc[start_idx:entry_idx + 1],
            close=prices.iloc[start_idx:entry_idx + 1],
            window=window
        )
        atr_values = atr_indicator.average_true_range()
        local_vol = atr_values.iloc[-1] if len(atr_values) > 0 and not pd.isna(atr_values.iloc[-1]) else None
        
        if local_vol is None or np.isnan(local_vol) or local_vol <= 0:
            # Fallback to std if ATR calculation fails
            local_vol = prices.iloc[:entry_idx + 1].rolling(window=window).std().iloc[-1]
    else:
        # Fallback: use rolling standard deviation
        std = prices.rolling(window=window).std()
        local_vol = std.iloc[entry_idx] if entry_idx < len(std) else None
    
    if local_vol is None or np.isnan(local_vol):
        raise ValueError(f"Could not calculate local volatility at entry_idx={entry_idx}")

    # TP/SL in absolute price units (pips-style)
    tp = round(tp_mult * local_vol, 5)
    sl = round(sl_mult * local_vol, 5)

    for t in range(entry_idx + 1, end + 1):
        rt = prices.iloc[t] - p0
        if rt >= tp:
            return 1, t, rt
        if rt <= -sl:
            return -1, t, rt

    end_rt = prices.iloc[end] - p0

    return 0, end, end_rt


def ma_cross_entries(
    price: pd.Series,
    fast_ma_period: int = 10,
    slow_ma_period: int = 100,
) -> tuple[pd.Series, pd.Series]:
    """
    Returns:
      long_entries  (Series of 0/1) — entry at t when signal occurred at t-1 (look-ahead avoided)
      short_entries (Series of 0/1)
    """
    fast_ema = ta.trend.EMAIndicator(price, window=fast_ma_period).ema_indicator()
    slow_ema = ta.trend.EMAIndicator(price, window=slow_ma_period).ema_indicator()

    rel_pos = (fast_ema > slow_ema).astype(int)

    long_signal = np.where((rel_pos.shift(1) == 0) & (rel_pos == 1), 1, 0)
    short_signal = np.where((rel_pos.shift(1) == 1) & (rel_pos == 0), 1, 0)

    # IMPORTANT: shift(1) to execute after signal bar (avoid look-ahead)
    long_entries = pd.Series(long_signal, index=price.index).shift(1)
    short_entries = pd.Series(short_signal, index=price.index).shift(1)

    return long_entries, short_entries


def generate_trades(
    price_df: pd.DataFrame,
    H: int = 50,
    tp_mult: float = 1.5,
    sl_mult: float = 1.0,
    fast_ma_period: int = 10,
    slow_ma_period: int = 30,
    method: str = "atr",
    past_bars: int = 50,
    price_col: str = "close_price",
    high_col: str | None = None,
    low_col: str | None = None,
    side: str = "long",
    feature_window: int = 20,
    feature_cols: list[str] | None = None,
    lag_price_features: bool = True,  # Lag price columns by 1 period to avoid look-ahead bias
) -> list[dict]:
    """
    Replicates your current logic.

    price_df: DataFrame with a datetime index and a column price_col ('close_price').
              Optionally includes high_col and low_col for proper ATR calculation.
    side: currently only 'long' is implemented to match your code. ('short' can be added.)
    """
    if price_col not in price_df.columns:
        raise ValueError(f"price_df must include column '{price_col}'")

    prices = price_df[price_col].astype(float)
    
    # Handle missing values: forward fill then backward fill
    n_missing_before = prices.isna().sum()
    if n_missing_before > 0:
        prices = prices.ffill().bfill()
        n_missing_after = prices.isna().sum()
        if n_missing_after > 0:
            raise ValueError(f"Cannot fill all missing prices. Still have {n_missing_after} NaN values after forward/backward fill.")
    
    # Extract high and low prices if available
    high_prices = None
    low_prices = None
    if high_col and high_col in price_df.columns:
        high_prices = price_df[high_col].astype(float)
        if high_prices.isna().any():
            high_prices = high_prices.ffill().bfill()
    if low_col and low_col in price_df.columns:
        low_prices = price_df[low_col].astype(float)
        if low_prices.isna().any():
            low_prices = low_prices.ffill().bfill()
    
    long_entries, short_entries = ma_cross_entries(prices, fast_ma_period=fast_ma_period, slow_ma_period=slow_ma_period)

    trades: list[dict] = []

    if side not in {"long"}:
        raise NotImplementedError("Only side='long' is implemented to replicate your current script.")

    # positions where long_entries == 1
    long_entry_mask = long_entries == 1
    long_entry_positions = [i for i, val in enumerate(long_entry_mask) if bool(val)]

    for e in tqdm(long_entry_positions, desc="Generating trades", leave=False):
        if e >= len(prices) - 2:
            continue

        label, exit_idx, exit_rt = triple_barrier_labels(
            prices=prices,
            entry_idx=e,
            H=H,
            tp_mult=tp_mult,
            sl_mult=sl_mult,
            method=method,
            past_bars=past_bars,
            high_prices=high_prices,
            low_prices=low_prices,
        )

        # Compute features, handling cases where there isn't enough history
        price_in = prices.iloc[e]
        price_out = prices.iloc[exit_idx]
        
        # Skip if prices are NaN
        if pd.isna(price_in) or pd.isna(price_out):
            continue
        
        # Initialize feature dictionary with basic trade information
        feat = {
            "entry_idx": int(e),
            "exit_idx": int(exit_idx),
            "hold_bars": int(exit_idx - e),
            "price_in": float(price_in),
            "price_out": float(price_out),
            "ret_log": float(np.log(price_out / price_in)),
            "label": int(label),
            "exit_rt_abs": float(exit_rt),
            "side": "long",
        }
        
        # Extract feature columns from price_df if specified
        # We trust that feature_cols are already present in price_df
        if feature_cols is not None:
            for feat_col in feature_cols:
                # Skip features that are already computed above or will be added later
                if feat_col in ["hold_bars", "is_synth", "entry_idx", "exit_idx", "price_in", "price_out", "ret_log", "label", "exit_rt_abs", "side"]:
                    continue
                
                # Extract feature from price_df if it exists
                if feat_col in price_df.columns:
                    # Determine if this is a price column (needs lagging to avoid look-ahead bias)
                    price_columns = ["close_price", "high_price", "low_price", "bid_price_close", 
                                     "ask_price_close", "bid_price_high", "bid_price_low",
                                     "ask_price_high", "ask_price_low", price_col]
                    is_price_col = feat_col in price_columns or feat_col == high_col or feat_col == low_col
                    
                    # For price columns, use lagged value (e-1) to avoid look-ahead bias
                    # For other features, use current value (e)
                    if lag_price_features and is_price_col and e > 0:
                        feat_val = price_df[feat_col].iloc[e - 1]  # Lag by 1 period
                    elif lag_price_features and is_price_col and e == 0:
                        # Can't lag at index 0, use NaN
                        feat_val = np.nan
                    else:
                        feat_val = price_df[feat_col].iloc[e]  # Use current value for non-price features
                    
                    # Convert to float, handling NaN
                    if pd.isna(feat_val):
                        feat[feat_col] = np.nan
                    else:
                        feat[feat_col] = float(feat_val)
                else:
                    # Feature not found in price_df - set to NaN
                    feat[feat_col] = np.nan
        # breakpoint()
        trades.append(feat)

    return trades


def generate_trades_from_paths(
    paths_df: pd.DataFrame,
    time_col: str = "t",
    path_col: str = "path",
    price_col_in: str = "price",
    out_price_col: str = "close_price",
    **trade_kwargs,
) -> pd.DataFrame:
    """
    Helper to apply generate_trades() to a multi-path synthetic DataFrame of the form:
      columns: [path, t, price]
    It converts each path group into the expected format (datetime index optional),
    calls generate_trades, and returns a single trades DataFrame with 'path' attached.

    trade_kwargs are passed to generate_trades (H, tp_mult, fast_ma_period, etc.)
    """
    if path_col not in paths_df.columns or price_col_in not in paths_df.columns:
        raise ValueError(f"paths_df must include '{path_col}' and '{price_col_in}' columns")

    rows = []
    path_groups = list(paths_df.groupby(path_col))
    
    # Extract feature_cols and other parameters from trade_kwargs
    feature_cols = trade_kwargs.get('feature_cols', None)
    feature_window = trade_kwargs.get('feature_window', 20)
    fast_ma_period = trade_kwargs.get('fast_ma_period', 10)
    slow_ma_period = trade_kwargs.get('slow_ma_period', 30)
    
    for pid, g in tqdm(path_groups, desc="Processing synthetic paths", leave=False):
        g2 = g.sort_values(time_col).reset_index(drop=True)
        # Use an integer index as a surrogate; your logic uses iloc anyway
        df_price = pd.DataFrame({out_price_col: g2[price_col_in].astype(float).to_numpy()})
        # Give it an index (could be datetime if you have it; not required for iloc operations)
        df_price.index = pd.RangeIndex(start=0, stop=len(df_price), step=1)
        
        # Compute feature columns that are requested but not already in the data
        # For synthetic paths, we typically need to compute features since they're generated
        if feature_cols is not None:
            prices_series = df_price[out_price_col].astype(float)
            # Handle missing values
            n_missing_before = prices_series.isna().sum()
            if n_missing_before > 0:
                prices_series = prices_series.ffill().bfill()
            
            # Only compute features that are requested AND not already present
            for feat_col in feature_cols:
                # Skip if already in df_price
                if feat_col in df_price.columns:
                    continue
                # Skip features computed elsewhere
                if feat_col in ["hold_bars", "is_synth"]:
                    continue
                
                # Compute features that need to be calculated
                pct_change = prices_series.pct_change(fill_method=None)
                if feat_col == "mom20" or feat_col == "mom":
                    df_price["mom20"] = pct_change.rolling(feature_window, min_periods=feature_window).mean()
                elif feat_col == "vol20" or feat_col == "vol":
                    df_price["vol20"] = pct_change.rolling(feature_window, min_periods=feature_window).std()
                elif feat_col == "ma_ratio":
                    fast_ma = prices_series.rolling(fast_ma_period).mean()
                    slow_ma = prices_series.rolling(slow_ma_period).mean()
                    df_price["ma_ratio"] = fast_ma / (slow_ma + 1e-12)
                # Other features that need computation can be added here

        trades_list = generate_trades(df_price, price_col=out_price_col, **trade_kwargs)
        for tr in trades_list:
            tr["path"] = int(pid)
            rows.append(tr)

    return pd.DataFrame(rows)
