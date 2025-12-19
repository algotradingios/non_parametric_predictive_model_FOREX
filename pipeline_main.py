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
from qfin_synth.walkforward import walk_forward_splits, filter_trades_for_fold
from qfin_synth.validation import validate_simulator

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

    # walk-forward
    train_bars: int = 2500,
    test_bars: int = 500,
    step_bars: int | None = None,
    embargo: int | None = None,

    # synthetic usage
    rho_max: float = 2.0,      # max ratio synthetic:real trades in train (cap)
):
    if embargo is None:
        embargo = H  # sensible default given labeling horizon

    # 1) Load prices
    logger.info(f"Loading historical data from {historical_csv}")
    df_hist = pd.read_csv(historical_csv, index_col=time_col, parse_dates=True).sort_index()
    prices = df_hist[price_col].astype(float)
    n_bars = len(prices)
    logger.info(f"Loaded {n_bars} price bars from {df_hist.index[0]} to {df_hist.index[-1]}")

    # Detect high/low columns if available
    high_col = 'bid_price_high' if 'bid_price_high' in df_hist.columns else None
    low_col = 'bid_price_low' if 'bid_price_low' in df_hist.columns else None
    
    # Prepare price_df with available columns
    price_df_cols = {price_col: "close_price"}
    if high_col and high_col in df_hist.columns:
        price_df_cols[high_col] = "high_price"
    if low_col and low_col in df_hist.columns:
        price_df_cols[low_col] = "low_price"
    
    price_df = df_hist[list(price_df_cols.keys())].rename(columns=price_df_cols)

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
    )
    real_trades_all = pd.DataFrame(real_trades_list)
    if len(real_trades_all) == 0:
        raise ValueError("No real trades generated. Check strategy parameters and data.")
    logger.info(f"Generated {len(real_trades_all)} real trades")

    # 3) Walk-forward splits
    logger.info(f"Computing walk-forward splits (train_bars={train_bars}, test_bars={test_bars}, step_bars={step_bars})")
    splits = walk_forward_splits(n_bars=n_bars, train_bars=train_bars, test_bars=test_bars, step_bars=step_bars)
    logger.info(f"Created {len(splits)} walk-forward folds")

    fold_results = []

    for fold, (tr0, tr1, te0, te1) in enumerate(tqdm(splits, desc="Processing folds"), start=1):
        logger.info(f"Fold {fold}/{len(splits)}: Train [{tr0}:{tr1}], Test [{te0}:{te1}]")
        # 3a) Select train/test price segments
        prices_train = prices.iloc[tr0:tr1+1]
        prices_test = prices.iloc[te0:te1+1]

        # 3b) Filter REAL trades for this fold (no leakage by exit_idx embargo)
        train_real, test_real = filter_trades_for_fold(
            trades=real_trades_all,
            train_start=tr0, train_end=tr1,
            test_start=te0, test_end=te1,
            embargo=embargo,
        )
        # Binary setup: drop label==0, map {1,-1}->{1,0}
        train_real = train_real[train_real["label"].isin([1, -1])].copy()
        test_real = test_real[test_real["label"].isin([1, -1])].copy()
        train_real["label_bin"] = (train_real["label"] == 1).astype(int)
        test_real["label_bin"] = (test_real["label"] == 1).astype(int)
        train_real["is_synth"] = 0
        test_real["is_synth"] = 0

        if len(train_real) < 50 or len(test_real) < 20:
            # Fold too thin; skip or relax parameters
            logger.warning(f"Fold {fold} skipped: Too few trades (train={len(train_real)}, test={len(test_real)})")
            fold_results.append({
                "fold": fold, "train_range": (tr0,tr1), "test_range": (te0,te1),
                "skipped": True, "reason": "Too few trades in train/test after purging."
            })
            continue

        logger.info(f"Fold {fold}: {len(train_real)} train trades, {len(test_real)} test trades")

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
        est = MuSigmaNonParam(h_mu=h_mu, h_sigma=h_sigma).fit(X, y)
        logger.info(f"Fold {fold}: Fitted estimator on {len(X)} training samples")

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
            synth_trades = generate_trades_from_paths(
                synth_prices,
                time_col="t",
                path_col="path",
                price_col_in="price",
                out_price_col="close_price",
                H=H, tp_mult=tp_mult, sl_mult=sl_mult,
                fast_ma_period=fast_ma_period, slow_ma_period=slow_ma_period,
                method=method, past_bars=past_bars,
                side="long",
            )
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
        logger.info(f"Fold {fold}: Training classifier ({len(train_df)} samples)...")
        train_df = pd.concat([train_real, synth_trades], ignore_index=True) if use_synth else train_real.copy()
        train_df = train_df.rename(columns={"label_bin": "label"})
        test_df = test_real.rename(columns={"label_bin": "label"})

        clf, metrics = train_classifier(
            train_df,
            feature_cols=["mom20", "vol20", "ma_ratio", "hold_bars", "is_synth"],
            test_size=0.2,  # internal split only for quick sanity; NOT the final evaluation
            seed=42,
        )

        # 9) True walk-forward evaluation: predict on test_df
        logger.info(f"Fold {fold}: Evaluating on test set ({len(test_df)} samples)...")
        X_test = test_df[["mom20", "vol20", "ma_ratio", "hold_bars", "is_synth"]].to_numpy(dtype=float)
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


def main(cfg: AppConfig):
    logger.info("=" * 80)
    logger.info("Starting pipeline execution")
    logger.info("=" * 80)
    results = run_pipeline_walkforward(
        historical_csv=cfg.historical_csv,
        time_col=cfg.time_col,
        price_col=cfg.price_col,

        # walk-forward
        train_bars=cfg.train_bars,
        test_bars=cfg.test_bars,
        step_bars=cfg.step_bars,
        embargo=cfg.embargo,

        # trades
        H=cfg.H,
        tp_mult=cfg.tp_mult,
        sl_mult=cfg.sl_mult,
        fast_ma_period=cfg.fast_ma_period,
        slow_ma_period=cfg.slow_ma_period,
        method=cfg.method,
        past_bars=cfg.past_bars,

        # state / simulator
        vol_window=cfg.vol_window,
        alpha=cfg.alpha,
        warmup=cfg.warmup,
        h_mu=cfg.h_mu,
        h_sigma=cfg.h_sigma,
        n_paths=cfg.n_paths,
        n_steps=cfg.n_steps,
        dt=cfg.dt,
        burnin=cfg.burnin,

        # synthetic usage
        rho_max=cfg.rho_max,
    )

    # Summarize
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
            "simulator_ok": r["sim_diag"]["simulator_ok"],
            "ks_ret_stat": r["sim_diag"]["ks_ret_stat"],
            "ks_rv_stat": r["sim_diag"]["ks_rv_stat"],
        })

    summary_df = pd.DataFrame(rows)
    logger.info(f"Saving results to {cfg.out_summary_csv} and {cfg.out_full_json}")
    summary_df.to_csv(cfg.out_summary_csv, index=False)

    with open(cfg.out_full_json, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    logger.info("=" * 80)
    logger.info("Pipeline execution completed successfully")
    logger.info("=" * 80)
    print(f"Saved: {cfg.out_summary_csv}, {cfg.out_full_json}")
    if len(summary_df):
        logger.info("Summary statistics:")
        print(summary_df[["auc_test", "f1_test", "mcc_test"]].mean().to_string())


if __name__ == "__main__":
    import json
    import pandas as pd

    cfg = AppConfig()  # reads .env by default
    main(cfg)