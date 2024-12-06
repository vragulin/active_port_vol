# Test sim_one_port.py

import pytest as pt
import numpy as np
from src.hist_returns.sim_one_port import gen_port_rets, Utility, calc_port_path_stats

# Global Parameters
RTOL = 1e-6


@pt.fixture
def stock_rets():
    return np.array([
        [0.2, 0, -0.1],
        [-0.1, 0.1, 0],
        [0.3, 0.15, 0],
        [0, -0.2, 0.2],
        [-0.09, 0.05, -0.05]
    ])


@pt.fixture
def weights():
    return np.array([0.2, 0.3, 0.5])


@pt.fixture
def sim_idx():
    return np.array([
        [0, 2, 2, 3],
        [3, 1, 0, 1],
        [1, 0, 4, 4],
        [3, 1, 0, 0],
        [0, 3, 4, 2]
    ])


def test_gen_port_rets(stock_rets, weights, sim_idx):
    port_rets = gen_port_rets(stock_rets, weights, sim_idx)
    expected = np.array([
        [-0.01, 0.105, 0.105, 0.04],
        [0.04, 0.01, -0.01, 0.01],
        [0.01, -0.01, -0.028, -0.028],
        [0.04, 0.01, -0.01, -0.01],
        [-0.01, 0.04, -0.028, 0.105]
    ])
    np.testing.assert_array_almost_equal(port_rets, expected)


@pt.mark.parametrize("param, expected", [
    (0, 1.092843636),
    (1, 1.091638671),
    (2, 1.090431682),
    (3, 1.089224495)
])
def test_port_stats_ce_wealth(param, expected, stock_rets, weights, sim_idx):
    port_rets = gen_port_rets(stock_rets, weights, sim_idx)
    u = Utility(param=param)
    stats = calc_port_path_stats(port_rets, u)
    actual = stats['w_ce']
    assert pt.approx(actual, abs=RTOL) == expected, f"Error: expected {expected}, got {actual}"