# Monte Carlo Risk Engine

Reproducible market-risk simulation for a static multi-asset portfolio. It uses correlated multivariate-normal log returns to simulate terminal prices, then calculates value at risk (VaR) and conditional VaR / expected shortfall (CVaR).

This is an educational risk-analysis tool, not investment advice. It does not account for liquidity, changing correlations, jumps, defaults, transaction costs, or model risk.

## Run

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python risk_engine.py example_portfolio.json --simulations 100000 --horizon-days 10 --seed 42
```

Use `--output report.json` to save a JSON report. The fixed `--seed` makes runs reproducible.

## Portfolio configuration

`example_portfolio.json` has one entry per asset:

- `prices`: current prices, positive.
- `positions`: signed quantities; use negatives for short positions.
- `annualized_return`: expected simple annual return.
- `annualized_volatility`: annual standard deviation of return.
- `correlation`: symmetric positive-semidefinite correlation matrix with a diagonal of ones.

The engine returns 95% and 99% VaR/CVaR by default. VaR is the loss quantile; CVaR is the mean loss at or beyond that threshold. Both are floored at zero for a positive-loss risk report.

## Assumptions

The engine converts annual inputs to daily log-return moments, scales these to the selected number of trading days (252 per year), draws correlated return scenarios, and revalues the portfolio without rebalancing.
