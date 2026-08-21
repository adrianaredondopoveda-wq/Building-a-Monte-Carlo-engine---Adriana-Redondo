"""Streamlit interface for the Monte Carlo Risk Engine."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from risk_engine import simulate


EXAMPLE_PATH = Path(__file__).with_name("example_portfolio.json")

st.set_page_config(page_title="Monte Carlo Risk Engine", page_icon="📉", layout="wide")
st.title("📉 Monte Carlo Risk Engine")
st.caption("Simulación educativa de riesgo de cartera. No constituye asesoramiento financiero.")


@st.cache_data
def example_config() -> dict:
    return json.loads(EXAMPLE_PATH.read_text(encoding="utf-8"))


def make_config(portfolio: pd.DataFrame, correlation: np.ndarray) -> dict:
    return {
        "assets": portfolio["Asset"].astype(str).tolist(),
        "prices": portfolio["Price"].astype(float).tolist(),
        "positions": portfolio["Position"].astype(float).tolist(),
        "annualized_return": portfolio["Expected annual return"].astype(float).tolist(),
        "annualized_volatility": portfolio["Annual volatility"].astype(float).tolist(),
        "correlation": correlation.tolist(),
    }


config = example_config()
default_portfolio = pd.DataFrame(
    {
        "Asset": config["assets"],
        "Price": config["prices"],
        "Position": config["positions"],
        "Expected annual return": config["annualized_return"],
        "Annual volatility": config["annualized_volatility"],
    }
)

with st.sidebar:
    st.header("Simulation settings")
    simulations = st.select_slider("Scenarios", options=[5_000, 10_000, 25_000, 50_000, 100_000, 200_000], value=50_000)
    horizon_days = st.slider("Horizon (trading days)", min_value=1, max_value=252, value=10)
    seed = st.number_input("Random seed", min_value=0, max_value=2_147_483_647, value=42, step=1)
    st.caption("Same inputs and seed produce the same result.")

st.subheader("Portfolio")
portfolio = st.data_editor(
    default_portfolio,
    num_rows="dynamic",
    use_container_width=True,
    column_config={
        "Price": st.column_config.NumberColumn(min_value=0.01, format="%.2f"),
        "Position": st.column_config.NumberColumn(format="%.2f"),
        "Expected annual return": st.column_config.NumberColumn(format="%.2f%%"),
        "Annual volatility": st.column_config.NumberColumn(min_value=0.0, format="%.2f%%"),
    },
    key="portfolio",
)

asset_count = len(portfolio)
if asset_count == 0:
    st.warning("Add at least one asset.")
    st.stop()

st.subheader("Correlation matrix")
default_correlation = np.eye(asset_count)
if asset_count == len(config["assets"]):
    default_correlation = np.asarray(config["correlation"], dtype=float)
correlation_frame = pd.DataFrame(default_correlation, index=portfolio["Asset"], columns=portfolio["Asset"])
correlation = st.data_editor(correlation_frame, use_container_width=True, key="correlation")

if st.button("Run simulation", type="primary"):
    try:
        result = simulate(
            make_config(portfolio, correlation.to_numpy(dtype=float)),
            simulations=int(simulations),
            horizon_days=int(horizon_days),
            seed=int(seed),
        )
    except (ValueError, TypeError) as error:
        st.error(f"Invalid portfolio configuration: {error}")
        st.stop()

    summary = result.summary()
    st.subheader("Risk summary")
    metrics = st.columns(4)
    metrics[0].metric("Portfolio value", f"{summary['initial_value']:,.0f}")
    metrics[1].metric("VaR 95%", f"{summary['var']['0.95']:,.0f}")
    metrics[2].metric("CVaR 95%", f"{summary['cvar']['0.95']:,.0f}")
    metrics[3].metric("Probability of loss", f"{summary['loss_probability']:.1%}")

    chart, stats = st.columns((2, 1))
    with chart:
        st.subheader("Simulated P&L distribution")
        counts, edges = np.histogram(result.pnl, bins=60)
        histogram = pd.DataFrame({"P&L": edges[:-1], "Scenarios": counts}).set_index("P&L")
        st.bar_chart(histogram)
    with stats:
        st.subheader("Scenario statistics")
        st.write(f"Mean P&L: **{summary['mean_pnl']:,.0f}**")
        st.write(f"Worst scenario: **{summary['worst_pnl']:,.0f}**")
        st.write(f"Best scenario: **{summary['best_pnl']:,.0f}**")
        st.write(f"VaR 99%: **{summary['var']['0.99']:,.0f}**")
        st.write(f"CVaR 99%: **{summary['cvar']['0.99']:,.0f}**")

    st.download_button(
        "Download risk report (JSON)",
        data=json.dumps(summary, indent=2),
        file_name="risk_report.json",
        mime="application/json",
    )
