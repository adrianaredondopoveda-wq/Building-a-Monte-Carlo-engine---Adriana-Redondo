import json
from pathlib import Path
import unittest

import numpy as np

from risk_engine import load_config, simulate


CONFIG = Path(__file__).with_name("example_portfolio.json")


class RiskEngineTests(unittest.TestCase):
    def test_simulation_is_reproducible(self):
        config = load_config(CONFIG)
        first = simulate(config, simulations=2_000, horizon_days=10, seed=7)
        second = simulate(config, simulations=2_000, horizon_days=10, seed=7)
        self.assertTrue(np.array_equal(first.pnl, second.pnl))
        self.assertEqual(first.summary(), second.summary())

    def test_zero_volatility_has_no_loss_risk(self):
        config = json.loads(CONFIG.read_text())
        config["annualized_return"] = [0.0, 0.0, 0.0]
        config["annualized_volatility"] = [0.0, 0.0, 0.0]
        result = simulate(config, simulations=100, horizon_days=10, seed=1)
        self.assertTrue(np.allclose(result.pnl, 0.0))
        self.assertEqual(result.var[0.95], 0.0)
        self.assertEqual(result.cvar[0.99], 0.0)
