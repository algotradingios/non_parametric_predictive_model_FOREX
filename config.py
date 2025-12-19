# config.py
from __future__ import annotations

from typing import Optional, Literal
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class AppConfig(BaseSettings):
    """
    Configuration loaded from environment variables and/or a .env file.

    Usage:
        cfg = AppConfig()                 # reads .env by default
        cfg = AppConfig(_env_file="...")  # override env file path
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # -----------------------
    # Data I/O
    # -----------------------
    historical_csv: str = Field(..., description="Path to historical CSV file")
    time_col: str = Field("time", description="Datetime column name")
    price_col: str = Field("close_price", description="Price column name")
    parse_dates: bool = Field(True, description="Parse time_col as datetime")
    timezone: Optional[str] = Field(None, description="Optional timezone name (e.g., 'Europe/Madrid')")

    # -----------------------
    # Walk-forward
    # -----------------------
    train_bars: int = Field(2500, ge=100)
    test_bars: int = Field(500, ge=50)
    step_bars: Optional[int] = Field(None, ge=1, description="If None, defaults to test_bars inside code")
    embargo: Optional[int] = Field(None, ge=0, description="If None, defaults to H")

    # -----------------------
    # Trades / labeling
    # -----------------------
    H: int = Field(50, ge=1, description="Triple-barrier horizon (bars)")
    tp_mult: float = Field(1.5, gt=0)
    sl_mult: float = Field(1.0, gt=0)
    fast_ma_period: int = Field(10, ge=2)
    slow_ma_period: int = Field(30, ge=3)
    method: Literal["atr", "std"] = Field("atr")
    past_bars: int = Field(50, ge=2)
    side: Literal["long"] = Field("long")  # keep aligned with your current implementation

    # -----------------------
    # State / nonparametric estimator
    # -----------------------
    vol_window: int = Field(60, ge=5)
    alpha: Optional[float] = Field(None, gt=0, lt=1, description="EWMA alpha; if None uses 2/(vol_window+1)")
    warmup: Optional[int] = Field(None, ge=5, description="Warmup bars for initial EWMA variance; if None=vol_window")
    h_mu: float = Field(0.2, gt=0, description="Kernel bandwidth for mu (after scaling)")
    h_sigma: float = Field(0.2, gt=0, description="Kernel bandwidth for sigma (after scaling)")

    # -----------------------
    # Simulation
    # -----------------------
    n_paths: int = Field(500, ge=1)
    n_steps: int = Field(2000, ge=50)
    dt: float = Field(1.0, gt=0)
    burnin: int = Field(300, ge=0)
    seed: int = Field(123, ge=0)

    # -----------------------
    # Synthetic usage control
    # -----------------------
    rho_max: float = Field(2.0, gt=0, description="Cap synthetic trades to rho_max * real trades in each fold")

    # -----------------------
    # Outputs
    # -----------------------
    out_summary_csv: str = Field("walkforward_summary.csv")
    out_full_json: str = Field("walkforward_full_results.json")

    def resolved_step_bars(self) -> int:
        return self.step_bars if self.step_bars is not None else self.test_bars

    def resolved_embargo(self) -> int:
        return self.embargo if self.embargo is not None else self.H

