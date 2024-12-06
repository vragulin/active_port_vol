""" Demonstrate that undiversified portfolios result in excess vol vs. the index
    Use historical returns instead of the the covariance matrix
    V. Ragulin - 11-Sep-2023
"""

import numpy as np
import pandas as pd
import pickle
from pprint import pprint
from sys import exit
from pathlib import Path
import codetiming
from src.excess_vol_w_median import gen_sim_ports
from src.hist_returns.utility import Utility
from src.hist_returns.sim_one_port import gen_port_rets, calc_port_path_stats

DATA_DIR = r"../../data_5y"
BASKET = 'b500_basket_20231106.csv'
INDEX_HIST = 'idx_rets_20231106_vals.csv'
RETURNS_DATA = 'returns_20231106.pickle'
DAILY, WEEKLY = range(2)
SIM_FREQ = WEEKLY
AVERAGE_VARS = False  # Whether to average variances or volatilities
MEDIAN = False  # Use median rather than mean for vol estimation

# Portfolios
REGENERATE_PORTS = True
PORTS = 'portfolios.pickle'
RNG = np.random.default_rng(123)

N_PORTS = 10000
STOCKS_IN_PORT = 5
N_PATHS = 1000
CRRA = 2
TARGET_RET = 0.1138  # Annual return over the period

# Portfolio weight and probability of selection method
EQ_W, MCAP_W = range(2)
EQ_P, MCAP_P = range(2)


def load_data(data_dir: str, basket: str, returns: str) -> dict:
    data_dict = {}

    # Load stock weights
    data_dict['basket'] = pd.read_csv(Path(data_dir) / basket, index_col=0)

    # Load stock and index history data
    data_dict['rets'] = pickle.load(open(Path(data_dir) / returns, 'rb'))

    return data_dict


def adj_returns_to_target(stock_rets: pd.DataFrame, weights: pd.Series, target_ret: float) -> pd.DataFrame:
    """ Adjust daily returns so that the index return over the period meets the target return
    :param stock_rets: DataFrame with stock returns (n_dates x n_stocks)
    :param weights: index weights
    :param target_ret: target annual return
    :return: DataFrame with adjusted returns
    """
    port_rets = stock_rets @ weights
    port_final_val = np.prod(1 + port_rets)
    steps = len(stock_rets)
    years = (stock_rets.index[-1] - stock_rets.index[0]).days * steps / (steps-1) / 365.25
    target_final_val = (1 + target_ret) ** years
    adj_factor = (target_final_val / port_final_val) ** (1 / len(port_rets))
    stock_rets_adj = (1 + stock_rets) * adj_factor - 1
    return stock_rets_adj


def an_port_set(ports: dict, inp: dict, method: int = MCAP_W, scale_factor: float = 1,
                average_vars: bool = False, use_median: bool = False,
                n_paths: int = 10, CRRA: float = 1) -> dict:
    """ Analyse a set of portfolios, calculate portfolilo return distribution by bootstrapping his prices
    :param ports: container with the portfolio set details
    :param inp: container with market and index basket data
    :param method: if EQ_W we build equal weighted portfolios, else market-cap weighted
    :param scale_factor: used to translate weights into position size
    :param average_vars: whether to average vars or vols
    :param use_median: whether to use median (not mean) for averaging
    :param n_paths: number of scenario paths for each portfolio
    :param CRRA: coefficient of relative risk aversion
    :return: dict with portfolio analytics
    """

    port_indices = ports['ports_indices']
    n_ports, stks_in_port = port_indices.shape
    idx_weights = inp['basket']['Percent Weight']
    idx_weights /= idx_weights.sum()

    if method == MCAP_W:
        # Generate portfolio positions
        def pos_from_idx(i: int):
            return idx_weights.values[i]

        pos_weights = np.vectorize(pos_from_idx)(port_indices) * scale_factor

    else:
        pos_weights = np.ones((n_ports, stks_in_port)) / stks_in_port * scale_factor

    # Calculate full portfolio weights
    n_stocks = len(inp['basket'])
    ports_full = np.zeros((n_ports, n_stocks))
    for i in range(n_stocks):
        ports_full[:, i] = (pos_weights * (port_indices == i)).sum(axis=1)
    ports_full_norm = (ports_full / ports_full.sum(axis=1)[:, None])  # normalized weights adding to 1

    # Compare aggregate weight vs. the index
    w_agg = ports_full.sum(axis=0)
    df_agg = pd.DataFrame(np.nan, index=idx_weights.index, columns=['w_agg', 'w_tgt', 'd_w'])
    df_agg['w_agg'] = w_agg / w_agg.sum()
    df_agg['w_tgt'] = idx_weights
    df_agg['d_w'] = df_agg.w_agg - df_agg.w_tgt

    # Calculate the distribution of portfolio returns using the bootstrapping method
    # for the protfolios and the index

    # Assuming inp['rets']['r_stk_filled'] is a DataFrame with daily returns
    daily_returns = inp['rets']['r_stk_filled']

    if SIM_FREQ == DAILY:
        n_steps = 252
        stock_rets_raw = inp['rets']['r_stk_filled']
    elif SIM_FREQ == WEEKLY:
        n_steps = 52
        weekly_rets_rolling = np.exp(np.log(1 + daily_returns).rolling(5).sum()) - 1
        stock_rets_raw = weekly_rets_rolling.iloc[4::5]
    else:
        raise ValueError("Invalid simulation frequency")

    stock_rets = adj_returns_to_target(stock_rets_raw, idx_weights, target_ret=TARGET_RET)

    scenarios = RNG.choice(n_steps, size=(n_steps, n_paths))
    util = Utility(param=CRRA)

    port_exp_rets = np.zeros(n_ports)
    port_ce_rets = np.zeros(n_ports)
    port_stds = np.zeros(n_ports)

    for i, port in enumerate(ports_full_norm):
        port_ret_paths = gen_port_rets(stock_rets.values, port, scenarios)
        stats = calc_port_path_stats(port_ret_paths, util)
        port_exp_rets[i] = stats['exp_ret']
        port_ce_rets[i] = stats['exp_ret_ce']
        port_stds[i] = stats['w_std']

    idx_ret_paths = gen_port_rets(stock_rets.values, idx_weights.values, scenarios)
    idx_stats = calc_port_path_stats(idx_ret_paths, util)

    # Pack results into output dict
    return {'pos_weights': pos_weights,
            'ports_full': ports_full,
            'w_stats': df_agg,
            'port_exp_rets': port_exp_rets,
            'port_ce_rets': port_ce_rets,
            'port_stds': port_stds,
            'idx_exp_ret': idx_stats['exp_ret'],
            'idx_ce_ret': idx_stats['exp_ret_ce'],
            'idx_std': idx_stats['w_std']
            }


if __name__ == "__main__":
    # Load data
    inp = load_data(data_dir=DATA_DIR, basket=BASKET, returns=RETURNS_DATA)

    # generate portfolios
    if REGENERATE_PORTS:
        ports = {}

        # Stock inclusion is proportional to market cap
        weights = inp['basket']['Percent Weight']
        ports[MCAP_P] = gen_sim_ports(N_PORTS, STOCKS_IN_PORT, weights)

        # Equal probability of inclusio for each stock
        weights = weights * 0 + 1 / len(weights)
        ports[EQ_P] = gen_sim_ports(N_PORTS, STOCKS_IN_PORT, weights)

        pickle.dump(ports, file=open(str(Path(DATA_DIR) / PORTS), 'wb'))
    else:
        ports = pickle.load(file=open(str(Path(DATA_DIR) / PORTS), 'rb'))

    print(f"\nSim settings: # stocks {STOCKS_IN_PORT}, # ports {N_PORTS}\n")

    # Base case - equal probability, market-cap weight
    with codetiming.Timer("an_port_set execution"):
        res = an_port_set(ports[EQ_P], inp, method=MCAP_W, n_paths=N_PATHS, CRRA=CRRA)

    print("\nEqual Prob, Market-Cap weighted Portfolios:")
    # pprint(res)
    print("Avg Port Vol = {:.3f}, Avg Basket Vol = {:.3f}".format(
        res['port_stds'].mean(), res['idx_std']))
    print("Avg Port Exp Ret = {:.3f}, Avg Basket Exp Ret = {:.3f}".format(
        res['port_exp_rets'].mean(), res['idx_exp_ret']))
    print("Avg Port CE Ret = {:.3f}, Avg Basket CE Ret = {:.3f}".format(
        res['port_ce_rets'].mean(), res['idx_ce_ret']))
    exit(0)
