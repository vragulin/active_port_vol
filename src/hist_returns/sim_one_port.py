""" Run a simulation for a single portfolio, calculate its certainty-equivalent return
    V. Ragulin - 12-Apr-2023
"""
import numpy as np
import pprint
import sys
from src.hist_returns.utility import Utility


def gen_port_rets(stock_rets: np.ndarray, weights: np.ndarray, sim_idx: np.ndarray) -> np.ndarray:
    """
    Simulate a single portfolio
    :param stock_rets: txn array of period returns for each stock
    :param weights: stock weights - nx1 vector, assume we rebalance at the end of each period
    :param sim_idx: txs array of dates indices to use for each simulation
    :return: dictionary with portfolio returns and statistics
    """

    # Calculate portfolio returns on consecutive dates
    port_rets_actual = stock_rets @ weights

    # Generate portfoolio returns across different paths
    port_rets_paths = port_rets_actual[sim_idx]

    return port_rets_paths


def calc_port_path_stats(port_ret_paths: np.ndarray, util: Utility) -> dict:
    """ Calculate summary statistics for a single portfolio simulation paths
        :param port_ret_paths: numpy array of portfolio returns for different simulation paths
        :param util: Utility object to calculate certainty-equivalent return, or None
        :return: dictionary with summary statistics including mean, standard deviation, and certainty-equivalent return
    """

    # Calculate statistics
    final_vals = np.prod(1 + port_ret_paths, axis=0)
    exp_ret = np.mean(final_vals) - 1
    exp_lret = np.mean(np.log(final_vals))
    if util is not None:
        final_vals_ce = util.ce_wealth(final_vals)
        exp_ret_ce = final_vals_ce - 1
    else:
        final_vals_ce = exp_ret_ce = None

    return {'w_mean': np.mean(final_vals),
            'w_std': np.std(np.log(final_vals)),
            'w_ce': final_vals_ce,
            'exp_ret': exp_ret,
            'exp_lret': exp_lret,
            'exp_ret_ce': exp_ret_ce
            }


# Usage exmaple (see calculations in the docs/test.xlss spreadsheet)
if __name__ == "__main__":
    # Set up simulation parameters
    n_stocks = 3
    n_sim = 4
    stock_rets = np.array([
        [0.2, 0, -0.1],
        [-0.1, 0.1, 0],
        [0.3, 0.15, 0],
        [0, -0.2, 0.2],
        [-0.09, 0.05, -0.05]
    ])
    weights = np.array([0.2, 0.3, 0.5])
    sim_idx = np.array([
        [0, 2, 2, 3],
        [3, 1, 0, 1],
        [1, 0, 4, 4],
        [3, 1, 0, 0],
        [0, 3, 4, 2]
    ])

    # Generate a portfolio
    port_rets = gen_port_rets(stock_rets, weights, sim_idx)

    print("Portfolio returns:")
    print(port_rets)
    u = Utility(param=2)
    stats = calc_port_path_stats(port_rets, u)

    print("\nPortfolio Simulation Summary Statistics:")
    pprint.pprint(stats)

    sys.exit(0)
