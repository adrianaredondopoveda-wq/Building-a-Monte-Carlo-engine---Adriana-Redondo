#!/usr/bin/env python3
"""A compact, reproducible Monte Carlo market-risk engine.

It models asset log returns as multivariate normal, revalues a buy-and-hold
portfolio over a chosen horizon, and reports VaR and Expected Shortfall (CVaR).
It is intended for education and scenario analysis, not investment advice.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


TRADING_DAYS = 252


@dataclass(frozen=True)
class SimulationResult:
    initial_value: float
    terminal_values: np.ndarray
    pnl: np.ndarray
    var: dict[float, float]
    cvar: dict[float, float]

    def summary(self) -> dict[str, Any]:
        return {
            "initial_value": self.initial_value,
            "mean_pnl": float(np.mean(self.pnl)),
            "median_pnl": float(np.median(self.pnl)),
            "pnl_std": float(np.std(self.pnl, ddof=1)),
            "loss_probability": float(np.mean(self.pnl < 0)),
            "var": {str(k): v for k, v in self.var.items()},
            "cvar": {str(k): v for k, v in self.cvar.items()},
            "worst_pnl": float(np.min(self.pnl)),
            "best_pnl": float(np.max(self.pnl)),
        }


def _as_array(values: list[float], name: str) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1 or len(array) == 0:
        raise ValueError(f"{name} must be a non-empty one-dimensional array")
    return array


def validate_config(config: dict[str, Any]) -> None:
    required = {"assets", "prices", "positions", "annualized_return", "annualized_volatility", "correlation"}
    missing = required - config.keys()
    if missing:
        raise ValueError(f"Missing configuration fields: {', '.join(sorted(missing))}")
    size = len(config["assets"])
    if size == 0:
        raise ValueError("assets cannot be empty")
    for key in ("prices", "positions", "annualized_return", "annualized_volatility"):
        if len(config[key]) != size:
            raise ValueError(f"{key} must have {size} values")
    correlation = np.asarray(config["correlation"], dtype=float)
    if correlation.shape != (size, size):
        raise ValueError(f"correlation must be a {size}x{size} matrix")
    if not np.allclose(correlation, correlation.T, atol=1e-10):
        raise ValueError("correlation must be symmetric")
    if not np.allclose(np.diag(correlation), 1.0, atol=1e-10):
        raise ValueError("correlation diagonal must contain ones")
    if np.any(_as_array(config["prices"], "prices") <= 0):
        raise ValueError("prices must be positive")
    if np.any(_as_array(config["annualized_volatility"], "annualized_volatility") < 0):
        raise ValueError("annualized_volatility cannot be negative")
    if np.linalg.eigvalsh(correlation).min() < -1e-9:
        raise ValueError("correlation must be positive semidefinite")


def simulate(
    config: dict[str, Any],
    *,
    simulations: int = 100_000,
    horizon_days: int = 10,
    seed: int = 42,
    confidence_levels: tuple[float, ...] = (0.95, 0.99),
) -> SimulationResult:
    """Run a log-normal correlated-return simulation for a static portfolio."""
    validate_config(config)
    if simulations < 1 or horizon_days < 1:
        raise ValueError("simulations and horizon_days must be positive")
    if any(not 0 < level < 1 for level in confidence_levels):
        raise ValueError("confidence levels must lie strictly between 0 and 1")

    prices = _as_array(config["prices"], "prices")
    positions = _as_array(config["positions"], "positions")
    annual_return = _as_array(config["annualized_return"], "annualized_return")
    annual_volatility = _as_array(config["annualized_volatility"], "annualized_volatility")
    correlation = np.asarray(config["correlation"], dtype=float)

    # Arithmetic annual inputs are converted to daily log-return moments.
    daily_variance = annual_volatility**2 / TRADING_DAYS
    daily_mu = np.log1p(annual_return) / TRADING_DAYS - 0.5 * daily_variance
    covariance = correlation * np.outer(annual_volatility, annual_volatility) / TRADING_DAYS
    horizon_mu = daily_mu * horizon_days
    horizon_covariance = covariance * horizon_days

    rng = np.random.default_rng(seed)
    log_returns = rng.multivariate_normal(horizon_mu, horizon_covariance, size=simulations)
    terminal_prices = prices * np.exp(log_returns)
    initial_value = float(np.dot(prices, positions))
    terminal_values = terminal_prices @ positions
    pnl = terminal_values - initial_value
    losses = -pnl

    var: dict[float, float] = {}
    cvar: dict[float, float] = {}
    for level in confidence_levels:
        threshold = float(np.quantile(losses, level))
        var[level] = max(0.0, threshold)
        cvar[level] = max(0.0, float(np.mean(losses[losses >= threshold])))

    return SimulationResult(initial_value, terminal_values, pnl, var, cvar)


def load_config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def main() -> None:
    parser = argparse.ArgumentParser(description="Monte Carlo portfolio-risk engine")
    parser.add_argument("config", type=Path, help="Portfolio JSON configuration")
    parser.add_argument("--simulations", type=int, default=100_000)
    parser.add_argument("--horizon-days", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, help="Optional JSON output path")
    args = parser.parse_args()

    result = simulate(
        load_config(args.config),
        simulations=args.simulations,
        horizon_days=args.horizon_days,
        seed=args.seed,
    )
    report = result.summary()
    rendered = json.dumps(report, indent=2)
    print(rendered)
    if args.output:
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
