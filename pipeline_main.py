from __future__ import annotations
# pipeline_main.py (walk-forward version)
import logging
import numpy as np
import pandas as pd
from tqdm import tqdm
from config import AppConfig

from qfin_synth.state import compute_basic_state
from qfin_synth.nw import MuSigmaNonParam
from qfin_synth.sde import simulate_paths
from qfin_synth.trades import generate_trades, generate_trades_from_paths
from qfin_synth.models import train_classifier
from qfin_synth.walkforward import filter_trades_for_fold
from qfin_synth.validation import validate_simulator
from qfin_synth.tuning import tune_bandwidths

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def run_pipeline_walkforward(
    historical_csv: str,
    time_col: str = "time",
    price_col: str = "close_price",

    # state/simulator
    vol_window: int = 60,
    alpha: float | None = None,
    warmup: int | None = None,
    h_mu: float = 0.2,
    h_sigma: float = 0.2,
    n_paths: int = 500,        # empieza moderado por fold
    n_steps: int = 2000,
    dt: float = 1.0,
    burnin: int = 300,

    # trades
    H: int = 50,
    tp_mult: float = 1.5,
    sl_mult: float = 1.0,
    fast_ma_period: int = 10,
    slow_ma_period: int = 30,
    method: str = "atr",
    past_bars: int = 50,
    feature_window: int = 20,

    # train/test split
    train_split: float = 0.8,  # Fraction of data to use for training (0.8 = 80%)
    embargo: int | None = None,

    # synthetic usage
    rho_max: float = 2.0,      # max ratio synthetic:real trades in train (cap)
    
    # fold filtering
    min_train_trades: int = 50,
    min_test_trades: int = 20,
    
    # classifier features
    feature_cols: list[str] | str | None = None,
    read_frac: float = 0.1,   # just read 10% of the dataset in case it is too big

):
    if embargo is None:
        embargo = H  # sensible default given labeling horizon
    
    # Convert feature_cols from string to list if needed and filter out price columns
    if feature_cols is not None:
        if isinstance(feature_cols, str):
            feature_cols = [col.strip() for col in feature_cols.split(',') if col.strip()]
        elif not isinstance(feature_cols, list):
            feature_cols = []
        
        original_feature_cols = feature_cols.copy() if feature_cols else []
        
        # Filter out ONLY trade metadata columns (NOT price columns or computed features)
        # Price columns are allowed as features but will be lagged to avoid look-ahead bias
        excluded_cols = ["hold_bars", "is_synth",
                         "entry_idx", "exit_idx", "price_in", "price_out", "ret_log", 
                         "label", "exit_rt_abs", "side"]
        feature_cols = [col for col in feature_cols if col not in excluded_cols]
        
        if len(feature_cols) == 0:
            if len(original_feature_cols) > 0:
                filtered_out = [col for col in original_feature_cols if col in excluded_cols]
                logger.warning(
                    f"All provided feature columns were filtered out. Original: {original_feature_cols}, "
                    f"Filtered out: {filtered_out}. "
                    f"Using default features: ['mom20', 'vol20', 'ma_ratio']"
                )
                feature_cols = ["mom20", "vol20", "ma_ratio"]  # Fallback to defaults
            else:
                logger.warning("feature_cols is empty; using default features: ['mom20', 'vol20', 'ma_ratio']")
                feature_cols = ["mom20", "vol20", "ma_ratio"]  # Fallback to defaults
    else:
        raise ValueError("feature_cols cannot be None; must provide at least one feature column.")

    # 1) Load prices (only 10% of the data)
    logger.info(f"Loading historical data from {historical_csv}")
    # First, count total rows to calculate 10%
    total_rows = sum(1 for _ in open(historical_csv, 'r')) - 1
    rows_to_read = max(1, int(total_rows * read_frac))
    logger.info(f"Reading {rows_to_read} rows ({rows_to_read/total_rows*100:.1f}%) out of {total_rows} total rows")
    df_hist = pd.read_csv(historical_csv, index_col=time_col, parse_dates=True).sort_index().head(rows_to_read)
    
    # Detect and map price column if it doesn't exist with the expected name
    actual_price_col = price_col
    if price_col not in df_hist.columns:
        # Try common alternatives
        alternatives = ['bid_price_close', 'ask_price_close', 'close_price', 'close', 'price']
        for alt in alternatives:
            if alt in df_hist.columns:
                actual_price_col = alt
                logger.info(f"Column '{price_col}' not found, using '{actual_price_col}' instead")
                break
        else:
            raise ValueError(f"Price column '{price_col}' not found in CSV. Available columns: {list(df_hist.columns)}")
    
    prices = df_hist[actual_price_col].astype(float)
    n_bars = len(prices)
    logger.info(f"Loaded {n_bars} price bars from {df_hist.index[0]} to {df_hist.index[-1]}")

    # Detect high/low columns if available
    high_col = 'bid_price_high' if 'bid_price_high' in df_hist.columns else None
    low_col = 'bid_price_low' if 'bid_price_low' in df_hist.columns else None
    
    # Prepare price_df with available columns
    price_df_cols = {actual_price_col: "close_price"}
    if high_col and high_col in df_hist.columns:
        price_df_cols[high_col] = "high_price"
    if low_col and low_col in df_hist.columns:
        price_df_cols[low_col] = "low_price"
    
    # Add feature columns from df_hist if they exist
    # Price columns are allowed as features and will be lagged to avoid look-ahead bias
    if feature_cols is not None:
        for feat_col in feature_cols:
            # Skip features that are computed elsewhere (hold_bars, is_synth)
            excluded = ["hold_bars", "is_synth"]
            if feat_col in excluded:
                continue
            
            # Map price column names if they're requested as features
            price_col_mapping = {
                "close_price": "close_price",
                "high_price": "high_price", 
                "low_price": "low_price",
                "bid_price_close": actual_price_col if actual_price_col.startswith("bid") else None,
                "ask_price_close": "ask_price_close" if "ask_price_close" in df_hist.columns else None,
                "bid_price_high": high_col if high_col and high_col.startswith("bid") else None,
                "bid_price_low": low_col if low_col and low_col.startswith("bid") else None,
            }
            
            # If it's a price column that's already mapped, skip (already in price_df_cols)
            if feat_col in ["close_price", "high_price", "low_price"]:
                continue
            
            # If it's a price column that needs mapping
            if feat_col in price_col_mapping and price_col_mapping[feat_col]:
                source_col = price_col_mapping[feat_col]
                if source_col in df_hist.columns:
                    # Map to standard name if not already mapped
                    if feat_col == "bid_price_close" and actual_price_col in df_hist.columns:
                        continue  # Already mapped as close_price
                    elif feat_col in ["bid_price_high", "bid_price_low"]:
                        continue  # Already mapped as high_price/low_price
                    else:
                        price_df_cols[source_col] = feat_col
            
            # If feature exists in original data, include it
            elif feat_col in df_hist.columns:
                price_df_cols[feat_col] = feat_col

    price_df = df_hist[list(price_df_cols.keys())].rename(columns=price_df_cols)
    
    # Compute feature columns that are requested but not already in the data
    if feature_cols is not None:
        prices_series = price_df["close_price"].astype(float)
        # Handle missing values
        n_missing_before = prices_series.isna().sum()
        if n_missing_before > 0:
            prices_series = prices_series.ffill().bfill()
        
        # Only compute features that are requested AND not already present
        for feat_col in feature_cols:
            # Skip if already in price_df (extracted from data or mapped price columns)
            if feat_col in price_df.columns:
                continue
            # Skip features computed elsewhere
            if feat_col in ["hold_bars", "is_synth"]:
                continue
            
            # Compute features that need to be calculated
            pct_change = prices_series.pct_change(fill_method=None)
            if feat_col == "mom20" or feat_col == "mom":
                price_df["mom20"] = pct_change.rolling(feature_window, min_periods=feature_window).mean()
            elif feat_col == "vol20" or feat_col == "vol":
                price_df["vol20"] = pct_change.rolling(feature_window, min_periods=feature_window).std()
            elif feat_col == "ma_ratio":
                fast_ma = prices_series.rolling(fast_ma_period).mean()
                slow_ma = prices_series.rolling(slow_ma_period).mean()
                price_df["ma_ratio"] = fast_ma / (slow_ma + 1e-12)
            # Other features that need computation can be added here
        
        # Ensure price columns requested as features are available with their requested names
        # Map standard names to requested feature names if needed
        for feat_col in feature_cols:
            if feat_col == "close_price" and "close_price" not in price_df.columns:
                # Should already be there, but check
                pass
            elif feat_col == "bid_price_close" and "close_price" in price_df.columns:
                # Add alias if requested
                if "bid_price_close" not in price_df.columns:
                    price_df["bid_price_close"] = price_df["close_price"]
            elif feat_col == "high_price" and "high_price" not in price_df.columns:
                # Should already be there if high_col exists
                pass
            elif feat_col == "bid_price_high" and "high_price" in price_df.columns:
                if "bid_price_high" not in price_df.columns:
                    price_df["bid_price_high"] = price_df["high_price"]
            elif feat_col == "low_price" and "low_price" not in price_df.columns:
                # Should already be there if low_col exists
                pass
            elif feat_col == "bid_price_low" and "low_price" in price_df.columns:
                if "bid_price_low" not in price_df.columns:
                    price_df["bid_price_low"] = price_df["low_price"]

    # breakpoint()

    # 2) Precompute REAL trades once on full series (no leakage yet; we will filter per fold)
    logger.info("Generating real trades from historical data...")
    real_trades_list = generate_trades(
        price_df=price_df,
        H=H, tp_mult=tp_mult, sl_mult=sl_mult,
        fast_ma_period=fast_ma_period, slow_ma_period=slow_ma_period,
        method=method, past_bars=past_bars,
        price_col="close_price",
        high_col="high_price" if high_col else None,
        low_col="low_price" if low_col else None,
        side="long",
        feature_window=feature_window,
        feature_cols=feature_cols,
        lag_price_features=True,  # Lag price columns by 1 period to avoid look-ahead bias
    )
    real_trades_all = pd.DataFrame(real_trades_list)
    if len(real_trades_all) == 0:
        raise ValueError("No real trades generated. Check strategy parameters and data.")
    logger.info(f"Generated {len(real_trades_all)} real trades")

    # breakpoint()
    # 3) Simple train/test split (80% train, 20% test)
    train_split = 0.8
    train_end_idx = int(n_bars * train_split)
    tr0, tr1 = 0, train_end_idx - 1
    te0, te1 = train_end_idx, n_bars - 1
    
    logger.info(f"Using simple train/test split: Train [0:{tr1}] ({train_split*100:.0f}%), Test [{te0}:{te1}] ({(1-train_split)*100:.0f}%)")

    fold_results = []
    fold = 1  # Single fold
    
    logger.info(f"Fold {fold}: Train [{tr0}:{tr1}], Test [{te0}:{te1}]")
    # 3a) Select train/test price segments
    prices_train = prices.iloc[tr0:tr1+1]
    prices_test = prices.iloc[te0:te1+1]

    # breakpoint()
    # 3b) Filter REAL trades for this fold (no leakage by exit_idx embargo)
    train_real, test_real = filter_trades_for_fold(
        trades=real_trades_all,
        train_start=tr0, train_end=tr1,
        test_start=te0, test_end=te1,
        embargo=embargo,
    )
    logger.info(f"Fold {fold}: After filtering by fold - train: {len(train_real)} trades, test: {len(test_real)} trades")
    
    # Binary setup: drop label==0, map {1,-1}->{1,0}
    train_real = train_real[train_real["label"].isin([1, -1])].copy()
    test_real = test_real[test_real["label"].isin([1, -1])].copy()
    logger.info(f"Fold {fold}: After filtering labels - train: {len(train_real)} trades, test: {len(test_real)} trades")
    
    # Remove trades with NaN features (insufficient history or missing price data)
    if len(train_real) > 0:
        nan_counts_train = train_real[feature_cols].isna().sum()
        logger.info(f"Fold {fold}: Train trades NaN counts: {nan_counts_train.to_dict()}")
    if len(test_real) > 0:
        nan_counts_test = test_real[feature_cols].isna().sum()
        logger.info(f"Fold {fold}: Test trades NaN counts: {nan_counts_test.to_dict()}")
    
    train_real = train_real.dropna(subset=feature_cols + ["price_in"]).copy()
    test_real = test_real.dropna(subset=feature_cols + ["price_in"]).copy()
    logger.info(f"Fold {fold}: After dropna - train: {len(train_real)} trades, test: {len(test_real)} trades")
    
    train_real["label_bin"] = (train_real["label"] == 1).astype(int)
    test_real["label_bin"] = (test_real["label"] == 1).astype(int)
    train_real["is_synth"] = 0
    test_real["is_synth"] = 0

    if len(train_real) < min_train_trades or len(test_real) < min_test_trades:
        # Fold too thin; skip or relax parameters
        logger.warning(f"Fold {fold} skipped: Too few trades (train={len(train_real)}<{min_train_trades}, test={len(test_real)}<{min_test_trades})")
        fold_results.append({
            "fold": fold, "train_range": (tr0,tr1), "test_range": (te0,te1),
            "skipped": True, "reason": f"Too few trades in train/test after purging (train={len(train_real)}, test={len(test_real)})"
        })
        return fold_results

    logger.info(f"Fold {fold}: {len(train_real)} train trades, {len(test_real)} test trades")

    # breakpoint()
    # 4) Fit mu/sigma using ONLY training prices
    logger.info(f"Fold {fold}: Computing state and fitting non-parametric estimator...")
    df_for_state = pd.DataFrame({
        "time": prices_train.index,
        "price": prices_train.values,
    })
    df_state, X, y, meta = compute_basic_state(
        df_for_state, price_col="price", time_col="time",
        vol_window=vol_window, alpha=alpha, warmup=warmup
    )
    
    if len(X) == 0 or len(y) == 0:
        logger.warning(f"Fold {fold} skipped: No valid state samples after computation (X.shape={X.shape}, y.shape={y.shape}, train_bars={len(prices_train)})")
        fold_results.append({
            "fold": fold, "train_range": (tr0,tr1), "test_range": (te0,te1),
            "skipped": True, "reason": f"No valid state samples (X.shape={X.shape}, train_bars={len(prices_train)})"
        })
        return fold_results
    
    est = MuSigmaNonParam(h_mu=h_mu, h_sigma=h_sigma).fit(X, y)
    logger.info(f"Fold {fold}: Fitted estimator on {len(X)} training samples")

    #  breakpoint()

    # Decide whether synthetics are even allowed (baseline support)
    allow_synth = (rho_max > 0.0) and (n_paths is not None) and (n_paths > 0)

    if not allow_synth:
        logger.info(f"Fold {fold}: Synthetics disabled (rho_max={rho_max}, n_paths={n_paths}). Skipping simulation/validation.")
        sim_diag = {
            "simulator_ok": False,
            "ks_ret_stat": float("nan"),
            "ks_rv_stat": float("nan"),
        }
        use_synth = False
        synth_trades = pd.DataFrame()
    else:

        # 5) Simulate synthetic paths (train-only)
        logger.info(f"Fold {fold}: Simulating {n_paths} synthetic paths ({n_steps} steps each)...")
        synth_prices = simulate_paths(
            est=est,
            start_price=float(prices_train.iloc[-1]),
            n_paths=n_paths,
            n_steps=n_steps,
            dt=dt,
            ewma_alpha=meta["alpha"],
            burnin=burnin,
            init_sigma=meta["init_sigma_recommended"],
            seed=123 + fold,
        )

        logger.info(f"Fold {fold}: Generated {len(synth_prices) // n_steps} synthetic paths")

        # 6) Validate simulator BEFORE using synthetics
        logger.info(f"Fold {fold}: Validating simulator...")
        sim_diag = validate_simulator(
            real_prices=prices_train,
            synth_prices_df=synth_prices,
            price_col_synth="price",
            path_col="path",
            max_lag=20,
            vol_window=20,
        )

        # 7) Generate synthetic trades only if simulator passes
        synth_trades = pd.DataFrame()
        use_synth = bool(sim_diag["simulator_ok"])
        logger.info(f"Fold {fold}: Simulator validation: {'PASSED' if use_synth else 'FAILED'} "
                f"(KS ret={sim_diag.get('ks_ret_stat', 'N/A'):.3f}, "
                f"KS RV={sim_diag.get('ks_rv_stat', 'N/A'):.3f})")

        if use_synth:
            logger.info(f"Fold {fold}: Generating synthetic trades...")
            synth_trades = generate_trades_from_paths(synth_prices, feature_window=feature_window, feature_cols=feature_cols)
            if len(synth_trades) > 0:
                synth_trades = synth_trades[synth_trades["label"].isin([1, -1])].copy()
                synth_trades["label_bin"] = (synth_trades["label"] == 1).astype(int)
                synth_trades["is_synth"] = 1

                # Cap synthetic volume to rho_max * real trades (avoid dominating)
                cap = int(rho_max * len(train_real))
                if len(synth_trades) > cap:
                    logger.info(f"Fold {fold}: Capping synthetic trades from {len(synth_trades)} to {cap}")
                    synth_trades = synth_trades.sample(n=cap, random_state=42)
            logger.info(f"Fold {fold}: Generated {len(synth_trades)} synthetic trades")

    # 8) Train classifier on train_real (+ optional synth)
    train_df = pd.concat([train_real, synth_trades], ignore_index=True) if use_synth else train_real.copy()
    # Drop the old "label" column (with values 1/-1) and rename "label_bin" to "label" (with values 0/1)
    train_df = train_df.drop(columns=["label"]).rename(columns={"label_bin": "label"})
    test_df = test_real.drop(columns=["label"]).rename(columns={"label_bin": "label"})
    logger.info(f"Fold {fold}: Training classifier ({len(train_df)} samples)...")

    clf, metrics = train_classifier(
        train_df,
        feature_cols=feature_cols,
        test_size=0.2,  # internal split only for quick sanity; NOT the final evaluation
        seed=42,
    )

    # 9) Evaluate on test set
    logger.info(f"Fold {fold}: Evaluating on test set ({len(test_df)} samples)...")
    X_test = test_df[feature_cols].to_numpy(dtype=float)
    y_test = test_df["label"].astype(int).to_numpy()
    p = clf.predict_proba(X_test)[:, 1]
    yhat = (p >= 0.5).astype(int)

    # Compute fold metrics (manual, minimal)
    from sklearn.metrics import roc_auc_score, f1_score, matthews_corrcoef

    fold_eval = {
        "auc_test": float(roc_auc_score(y_test, p)),
        "f1_test": float(f1_score(y_test, yhat)),
        "mcc_test": float(matthews_corrcoef(y_test, yhat)),
    }
    logger.info(f"Fold {fold}: Test metrics - AUC={fold_eval['auc_test']:.4f}, "
               f"F1={fold_eval['f1_test']:.4f}, MCC={fold_eval['mcc_test']:.4f}")

    fold_results.append({
        "fold": fold,
        "train_range": (tr0, tr1),
        "test_range": (te0, te1),
        "n_train_real": int(len(train_real)),
        "n_train_synth": int(len(synth_trades)) if use_synth else 0,
        "n_test_real": int(len(test_real)),
        "used_synth": bool(use_synth),
        "sim_diag": sim_diag,
        "train_metrics_internal": metrics,
        "test_metrics_walkforward": fold_eval,
        "skipped": False,
    })

    return fold_results

def build_base_kwargs(cfg: AppConfig) -> dict:
    """
    Parámetros base para run_pipeline_walkforward (excepto h_mu/h_sigma).
    """
    # Process feature_cols before building the dict
    if hasattr(cfg, 'get_feature_cols'):
        feature_cols_val = cfg.get_feature_cols()
    else:
        feature_cols_val = getattr(cfg, "feature_cols", None)
        if isinstance(feature_cols_val, str):
            feature_cols_val = [col.strip() for col in feature_cols_val.split(',') if col.strip()]
    
    # Filter out ONLY trade metadata columns (NOT price columns or computed features)
    # Price columns are allowed as features but will be lagged to avoid look-ahead bias
    excluded_cols = ["hold_bars", "is_synth",
                     "entry_idx", "exit_idx", "price_in", "price_out", "ret_log", 
                     "label", "exit_rt_abs", "side"]
    
    if feature_cols_val:
        original_count = len(feature_cols_val)
        feature_cols_val = [col for col in feature_cols_val if col not in excluded_cols]
        if len(feature_cols_val) == 0 and original_count > 0:
            # All provided features were invalid, use defaults
            feature_cols_val = ["mom20", "vol20", "ma_ratio"]
    
    # Ensure we have a default if feature_cols is None or empty
    if not feature_cols_val or (isinstance(feature_cols_val, list) and len(feature_cols_val) == 0):
        feature_cols_val = ["mom20", "vol20", "ma_ratio"]  # Default features
    
    return dict(
        historical_csv=cfg.historical_csv,
        time_col=cfg.time_col,
        price_col=cfg.price_col,

        # state/simulator base
        vol_window=cfg.vol_window,
        alpha=cfg.alpha,
        warmup=cfg.warmup,
        n_paths=cfg.n_paths,
        n_steps=cfg.n_steps,
        dt=cfg.dt,
        burnin=cfg.burnin,

        # trades
        H=cfg.H,
        tp_mult=cfg.tp_mult,
        sl_mult=cfg.sl_mult,
        fast_ma_period=cfg.fast_ma_period,
        slow_ma_period=cfg.slow_ma_period,
        method=cfg.method,
        past_bars=cfg.past_bars,
        feature_window=getattr(cfg, "feature_window", 20),

        # train/test split (tu función usa split simple)
        train_split=getattr(cfg, "train_split", 0.8),
        embargo=cfg.embargo,

        # synthetic usage
        rho_max=cfg.rho_max,

        # fold filtering
        min_train_trades=getattr(cfg, "min_train_trades", 50),
        min_test_trades=getattr(cfg, "min_test_trades", 20),

        # classifier features
        feature_cols=feature_cols_val,

        # OPTIONAL (si aplicas el parche A.1)
        # read_frac=getattr(cfg, "read_frac", 1.0),
    )


def run_bandwidth_tuning(cfg: AppConfig) -> pd.DataFrame:
    base_kwargs = build_base_kwargs(cfg)

    # Grids pequeños iniciales
    h_mu_grid = [0.2, 0.3, 0.4]
    h_sigma_grid = [0.4, 0.6, 0.8]

    df_ranked = tune_bandwidths(
        run_pipeline_walkforward=run_pipeline_walkforward,
        base_kwargs=base_kwargs,
        h_mu_grid=h_mu_grid,
        h_sigma_grid=h_sigma_grid,
        score="mcc",
        prefer_robust=True,
    )

    df_ranked.to_csv("bandwidth_tuning_results.csv", index=False)

    cols = [
        "h_mu", "h_sigma", "eligible",
        "mean_mcc", "std_mcc", "delta_mcc",
        "mean_auc", "delta_auc",
        "used_synth_rate", "notes",
    ]
    print("\n=== BANDWIDTH TUNING (top 10) ===")
    print(df_ranked[cols].head(10).to_string(index=False))

    return df_ranked


def run_final_pipeline(cfg: AppConfig, best_h_mu: float, best_h_sigma: float) -> list[dict]:
    """
    Ejecuta una vez el pipeline con los mejores h.
    """
    kwargs = build_base_kwargs(cfg)
    kwargs["h_mu"] = float(best_h_mu)
    kwargs["h_sigma"] = float(best_h_sigma)

    results = run_pipeline_walkforward(**kwargs)

    # Resumen
    rows = []
    for r in results:
        if r.get("skipped"):
            continue
        m = r["test_metrics_walkforward"]
        rows.append({
            "fold": r["fold"],
            "train_start": r["train_range"][0],
            "train_end": r["train_range"][1],
            "test_start": r["test_range"][0],
            "test_end": r["test_range"][1],
            "used_synth": r["used_synth"],
            "n_train_real": r["n_train_real"],
            "n_train_synth": r["n_train_synth"],
            "n_test_real": r["n_test_real"],
            "auc_test": m["auc_test"],
            "f1_test": m["f1_test"],
            "mcc_test": m["mcc_test"],
            "simulator_ok": r.get("sim_diag", {}).get("simulator_ok", False),
            "ks_ret_stat": r.get("sim_diag", {}).get("ks_ret_stat", float("nan")),
            "ks_rv_stat": r.get("sim_diag", {}).get("ks_rv_stat", float("nan")),
        })

    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(cfg.out_summary_csv, index=False)

    with open(cfg.out_full_json, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\nSaved: {cfg.out_summary_csv}, {cfg.out_full_json}")
    if len(summary_df):
        print("\n=== FINAL (mean) ===")
        print(summary_df[["auc_test", "f1_test", "mcc_test"]].mean().to_string())

    return results

def main(cfg: AppConfig):
    logger.info("=" * 80)
    logger.info("Starting pipeline execution")
    logger.info("=" * 80)
    df_ranked = run_bandwidth_tuning(cfg)

    elig = df_ranked[df_ranked["eligible"] == True]
    if len(elig) > 0:
        best = elig.iloc[0]
        print("\nSelected best ELIGIBLE candidate.")
    else:
        best = df_ranked.iloc[0]
        print("\nWARNING: No eligible candidates. Selecting top-ranked candidate anyway.")

    best_h_mu = float(best["h_mu"])
    best_h_sigma = float(best["h_sigma"])

    with open("best_bandwidth.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "h_mu": best_h_mu,
                "h_sigma": best_h_sigma,
                "eligible": bool(best.get("eligible", False)),
                "mean_mcc": float(best.get("mean_mcc", float("nan"))),
                "delta_mcc": float(best.get("delta_mcc", float("nan"))),
                "std_mcc": float(best.get("std_mcc", float("nan"))),
                "used_synth_rate": float(best.get("used_synth_rate", float("nan"))),
                "notes": str(best.get("notes", "")),
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(f"\nBest h_mu={best_h_mu:.4f}, h_sigma={best_h_sigma:.4f}")
    run_final_pipeline(cfg, best_h_mu, best_h_sigma)

if __name__ == "__main__":
    import json
    import pandas as pd

    cfg = AppConfig()  # reads .env by default
    main(cfg)