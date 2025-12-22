# qfin_synth/tuning.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Optional

import numpy as np
import pandas as pd


@dataclass
class CandidateResult:
    h_mu: float
    h_sigma: float
    passed_stability: bool
    passed_validation: bool
    used_synth_rate: float  # fraction of folds where synthetics were used
    mean_auc: float
    mean_f1: float
    mean_mcc: float
    std_auc: float
    std_f1: float
    std_mcc: float
    mean_auc_nosynth: float
    mean_f1_nosynth: float
    mean_mcc_nosynth: float
    delta_auc: float  # mean_auc - mean_auc_nosynth
    delta_f1: float
    delta_mcc: float
    notes: str
    fold_details: list[dict]


def _safe_float(x, default=np.nan) -> float:
    try:
        return float(x)
    except Exception:
        return float(default)


def _aggregate_fold_metrics(fold_results: list[dict]) -> tuple[dict, float]:
    """
    Takes fold_results from run_pipeline_walkforward and aggregates only non-skipped folds.
    Returns (agg_metrics, used_synth_rate).
    """
    rows = []
    used_synth = []
    for r in fold_results:
        if r.get("skipped"):
            continue
        m = r.get("test_metrics_walkforward", {})
        rows.append(
            {
                "auc": _safe_float(m.get("auc_test")),
                "f1": _safe_float(m.get("f1_test")),
                "mcc": _safe_float(m.get("mcc_test")),
            }
        )
        used_synth.append(bool(r.get("used_synth", False)))

    if len(rows) == 0:
        return (
            {
                "mean_auc": np.nan, "mean_f1": np.nan, "mean_mcc": np.nan,
                "std_auc": np.nan, "std_f1": np.nan, "std_mcc": np.nan,
            },
            0.0,
        )

    df = pd.DataFrame(rows)
    used_synth_rate = float(np.mean(used_synth)) if used_synth else 0.0
    return (
        {
            "mean_auc": float(df["auc"].mean()),
            "mean_f1": float(df["f1"].mean()),
            "mean_mcc": float(df["mcc"].mean()),
            "std_auc": float(df["auc"].std(ddof=1)) if len(df) > 1 else 0.0,
            "std_f1": float(df["f1"].std(ddof=1)) if len(df) > 1 else 0.0,
            "std_mcc": float(df["mcc"].std(ddof=1)) if len(df) > 1 else 0.0,
        },
        used_synth_rate,
    )


def _check_simulator_health_from_folds(fold_results: list[dict]) -> tuple[bool, str]:
    """
    Stability+validation gate derived from per-fold sim_diag and basic sanity.
    If too many folds are skipped or sim fails everywhere, reject.
    """
    non_skipped = [r for r in fold_results if not r.get("skipped")]
    if len(non_skipped) == 0:
        return False, "All folds skipped (too few trades or split too tight)."

    # Count simulator_ok among non-skipped folds (it is computed each fold)
    oks = []
    for r in non_skipped:
        sim_diag = r.get("sim_diag", {})
        if "simulator_ok" in sim_diag:
            oks.append(bool(sim_diag["simulator_ok"]))
        else:
            oks.append(False)

    ok_rate = float(np.mean(oks)) if oks else 0.0
    if ok_rate == 0.0:
        return False, "Simulator validation failed in all non-skipped folds."
    # You may choose a stricter threshold (e.g. >= 0.5)
    if ok_rate < 0.25:
        return False, f"Simulator validation passed in too few folds (ok_rate={ok_rate:.2f})."
    return True, f"Simulator validation ok_rate={ok_rate:.2f}."


def tune_bandwidths(
    *,
    run_pipeline_walkforward: Callable[..., list[dict]],
    base_kwargs: dict,
    h_mu_grid: Iterable[float],
    h_sigma_grid: Iterable[float],
    score: str = "mcc",  # or "auc" / "f1"
    prefer_robust: bool = True,
) -> pd.DataFrame:
    """
    Grid-search over (h_mu, h_sigma) using the correct gating:
      1) stability/validation must pass (derived from fold sim_diag)
      2) choose by out-of-sample metrics in walk-forward
      3) compare to a no-synthetic baseline with identical folds/params

    Parameters
    ----------
    run_pipeline_walkforward:
        Your function that runs the full walk-forward and returns fold_results.
    base_kwargs:
        Dict with all kwargs for run_pipeline_walkforward except h_mu/h_sigma.
        It should include synthetic settings, walk-forward parameters, etc.
    h_mu_grid, h_sigma_grid:
        Candidate bandwidth values (post-scaling bandwidth).
    score:
        Which metric to rank by: "mcc" (recommended), "auc", or "f1".
    prefer_robust:
        If True, prefer higher mean score with lower std (tie-break).
    """
    results: list[CandidateResult] = []
    
    # Convert grids to lists to ensure they're finite and can be counted
    h_mu_grid = list(h_mu_grid)
    h_sigma_grid = list(h_sigma_grid)
    total_combinations = len(h_mu_grid) * len(h_sigma_grid)
    
    print(f"\nStarting bandwidth tuning: {len(h_mu_grid)} h_mu values × {len(h_sigma_grid)} h_sigma values = {total_combinations} combinations")
    print(f"Each combination runs twice (with/without synthetics) = {total_combinations * 2} total pipeline runs\n")

    # Baseline (no synthetics) is run per candidate to keep everything identical
    # except synthetic usage; this avoids confounding due to stochasticity.
    combination_num = 0
    for h_mu in h_mu_grid:
        for h_sigma in h_sigma_grid:
            combination_num += 1
            print(f"[{combination_num}/{total_combinations}] Testing h_mu={h_mu:.3f}, h_sigma={h_sigma:.3f}...")
            # --- WITH SYNTHETICS ---
            print(f"  Running with synthetics (h_mu={h_mu:.3f}, h_sigma={h_sigma:.3f})...")
            kwargs_synth = dict(base_kwargs)
            kwargs_synth["h_mu"] = float(h_mu)
            kwargs_synth["h_sigma"] = float(h_sigma)
            fold_synth = run_pipeline_walkforward(**kwargs_synth)
            print(f"  Completed with synthetics.")

            # Gate: simulator health from folds
            passed_validation, note = _check_simulator_health_from_folds(fold_synth)

            agg_synth, used_synth_rate = _aggregate_fold_metrics(fold_synth)

            # --- WITHOUT SYNTHETICS ---
            # Force no synthetics by setting rho_max=0 (or n_paths=0 if your code supports it).
            print(f"  Running without synthetics (h_mu={h_mu:.3f}, h_sigma={h_sigma:.3f})...")
            kwargs_nosynth = dict(base_kwargs)
            kwargs_nosynth["h_mu"] = float(h_mu)
            kwargs_nosynth["h_sigma"] = float(h_sigma)
            kwargs_nosynth["rho_max"] = 0.0
            # optionally also set n_paths small to save time; pipeline will not add synthetics anyway
            kwargs_nosynth["n_paths"] = 0
            fold_nosynth = run_pipeline_walkforward(**kwargs_nosynth)
            print(f"  Completed without synthetics.")
            agg_nosynth, _ = _aggregate_fold_metrics(fold_nosynth)

            # stability gate: if model collapses, often shows as NaN metrics or all folds skipped
            passed_stability = np.isfinite(agg_synth["mean_mcc"]) and not np.isnan(agg_synth["mean_mcc"])

            res = CandidateResult(
                h_mu=float(h_mu),
                h_sigma=float(h_sigma),
                passed_stability=bool(passed_stability),
                passed_validation=bool(passed_validation),
                used_synth_rate=float(used_synth_rate),
                mean_auc=float(agg_synth["mean_auc"]),
                mean_f1=float(agg_synth["mean_f1"]),
                mean_mcc=float(agg_synth["mean_mcc"]),
                std_auc=float(agg_synth["std_auc"]),
                std_f1=float(agg_synth["std_f1"]),
                std_mcc=float(agg_synth["std_mcc"]),
                mean_auc_nosynth=float(agg_nosynth["mean_auc"]),
                mean_f1_nosynth=float(agg_nosynth["mean_f1"]),
                mean_mcc_nosynth=float(agg_nosynth["mean_mcc"]),
                delta_auc=float(agg_synth["mean_auc"] - agg_nosynth["mean_auc"]),
                delta_f1=float(agg_synth["mean_f1"] - agg_nosynth["mean_f1"]),
                delta_mcc=float(agg_synth["mean_mcc"] - agg_nosynth["mean_mcc"]),
                notes=str(note),
                fold_details=fold_synth,  # keep the synth folds as details
            )
            results.append(res)
            print(f"  Result: eligible={bool(passed_stability and passed_validation)}, mean_mcc={agg_synth['mean_mcc']:.4f}\n")

    print(f"\nCompleted all {total_combinations} combinations. Ranking results...\n")
    
    # Convert to dataframe
    df = pd.DataFrame([r.__dict__ for r in results])

    # Ranking: only among those passing stability+validation
    df["eligible"] = df["passed_stability"] & df["passed_validation"]

    # Score selection
    score_col = {"auc": "mean_auc", "f1": "mean_f1", "mcc": "mean_mcc"}[score]
    std_col = {"auc": "std_auc", "f1": "std_f1", "mcc": "std_mcc"}[score]

    # Primary sort: eligible desc, score desc, delta_score desc, then robustness
    df["delta_score"] = df[score_col] - df[{"auc": "mean_auc_nosynth", "f1": "mean_f1_nosynth", "mcc": "mean_mcc_nosynth"}[score]]

    sort_cols = ["eligible", score_col, "delta_score"]
    ascending = [False, False, False]

    if prefer_robust:
        sort_cols += [std_col, "used_synth_rate"]
        ascending += [True, False]  # prefer lower std, higher usage

    df = df.sort_values(sort_cols, ascending=ascending).reset_index(drop=True)
    return df
