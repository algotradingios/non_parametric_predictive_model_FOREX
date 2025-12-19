# qfin_synth/walkforward.py
from __future__ import annotations
import numpy as np
import pandas as pd

def walk_forward_splits(
    n_bars: int,
    train_bars: int,
    test_bars: int,
    step_bars: int | None = None,
    min_train_bars: int | None = None,
):
    """
    Generates (train_start, train_end, test_start, test_end) in bar indices.
    - train is [train_start, train_end] inclusive
    - test  is [test_start, test_end] inclusive
    """
    if step_bars is None:
        step_bars = test_bars
    if min_train_bars is None:
        min_train_bars = train_bars

    splits = []
    train_end = min_train_bars - 1
    while True:
        test_start = train_end + 1
        test_end = min(test_start + test_bars - 1, n_bars - 1)
        train_start = max(0, train_end - train_bars + 1)

        if test_start >= n_bars:
            break

        splits.append((train_start, train_end, test_start, test_end))

        # move forward
        train_end = train_end + step_bars
        if train_end >= n_bars - 1:
            break

    return splits

def filter_trades_for_fold(
    trades: pd.DataFrame,
    train_start: int,
    train_end: int,
    test_start: int,
    test_end: int,
    embargo: int,
):
    """
    Avoid leakage due to label horizon:
    - Train trades must be fully resolved before test starts (exit_idx < test_start - embargo)
    - Test trades must have entry within [test_start, test_end] AND exit within [test_start, test_end + embargo] allowed,
      but for evaluation you typically require exit_idx <= test_end (strict), otherwise you evaluate incomplete labels.
    """
    # Train: entries inside train window AND exits before the embargo boundary
    train_cut = test_start - embargo
    train_mask = (
        (trades["entry_idx"] >= train_start) &
        (trades["entry_idx"] <= train_end) &
        (trades["exit_idx"] < train_cut)
    )

    # Test: entries inside test window AND exits within test window (strict)
    test_mask = (
        (trades["entry_idx"] >= test_start) &
        (trades["entry_idx"] <= test_end) &
        (trades["exit_idx"] <= test_end)
    )

    return trades.loc[train_mask].copy(), trades.loc[test_mask].copy()
