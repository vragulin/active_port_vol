""" Demonstrate that undiversified portfolios result in excess vol vs. the index
    Use historical returns instead of the the covariance matrix
    Run a grid of values for  a list of CRRA, STOCKS_IN_PORT inputs
    V. Ragulin - 11-Sep-2023
"""

import numpy as np
import pandas as pd
import pickle
from pprint import pprint
from sys import exit
from pathlib import Path
import codetiming
from typing import Tuple
from src.excess_vol_w_median import gen_sim_ports
from src.hist_returns.utility import Utility
from src.hist_returns.sim_one_port import gen_port_rets, calc_port_path_stats
from src.hist_returns.cert_equiv_ret import an_port_set, load_data, adj_returns_to_target

DATA_DIR = r"../../data_5y"
BASKET = 'b500_basket_20231106.csv'
INDEX_HIST = 'idx_rets_20231106_vals.csv'
RETURNS_DATA = 'returns_20231106.pickle'
DAILY, WEEKLY = range(2)
SIM_FREQ = WEEKLY

# Portfolios
REGENERATE_PORTS = True
PORTS = 'portfolios.pickle'
RNG = np.random.default_rng(123)

N_PORTS = 10000
STOCKS_IN_PORT_LIST = [5, 25, 100]
N_PATHS = 500
CRRA_LIST = [0, 1, 2, 5, 10]

TARGET_RET = 0.1138  # Annual return over the period

# Portfolio weight and probability of selection method
EQ_W, MCAP_W = range(2)
EQ_P, MCAP_P = range(2)


class Simulation:
    def __init__(self, data_dir: str, basket: str, returns: str, freq: int, n_paths: int,
                 target_ret: float | None = None):
        self.data_dir = data_dir
        self.inp = load_data(data_dir=data_dir, basket=basket, returns=returns)
        self.idx_weights = self.inp['basket']['Percent Weight']
        self.idx_weights /= self.idx_weights.sum()
        self.n_stocks = len(self.idx_weights)
        self.freq = freq
        self.n_steps = 52 if freq == WEEKLY else 252
        self.n_paths = n_paths
        self.sim_index = RNG.choice(self.n_steps, size=(self.n_steps, self.n_paths))

        # Prepare stock returns to match the target
        daily_returns = self.inp['rets']['r_stk_filled']

        if SIM_FREQ == DAILY:
            stock_rets_raw = daily_returns
        elif SIM_FREQ == WEEKLY:
            weekly_rets_rolling = np.exp(np.log(1 + daily_returns).rolling(5).sum()) - 1
            stock_rets_raw = weekly_rets_rolling.iloc[4::5]
        else:
            raise ValueError("Invalid simulation frequency")

        if target_ret is not None:
            self.stock_rets = adj_returns_to_target(stock_rets_raw, self.idx_weights, target_ret=target_ret)
        else:
            self.stock_rets = stock_rets_raw

    def an_port_set(self, ports: dict, CRRA_list: list) -> dict:
        """ Analyze a set of portfolios with different CRRA values
        :param ports: dictionary with portfolio weights
        :param CRRA_list: list of CRRA values
        :return: dictionary with portfolio and index statistics
        """
        port_indices = ports['ports_indices']
        n_ports, stks_in_port = port_indices.shape
        stock_rets, idx_weights = self.stock_rets.values, self.idx_weights.values

        res = {
            'port_exp_rets': np.zeros(n_ports),
            'port_stds': np.zeros(n_ports),
            'port_ce_rets': np.zeros((len(CRRA_list), n_ports)),
            'idx_exp_ret': None,
            'idx_std': None,
            'idx_ce_ret': np.zeros(len(CRRA_list)),
        }

        # Generate portfolio positions and weights
        def pos_from_idx(pos_idx: int):
            return idx_weights[pos_idx]

        pos_weights = np.vectorize(pos_from_idx)(port_indices)
        ports_full = np.zeros((n_ports, self.n_stocks))
        for i in range(self.n_stocks):
            ports_full[:, i] = (pos_weights * (port_indices == i)).sum(axis=1)
        ports_full_norm = (ports_full / ports_full.sum(axis=1)[:, None])  # normalized weights adding to 1

        utils = [Utility(param=cr) for cr in CRRA_list]
        for j, port in enumerate(ports_full_norm):
            port_rets = gen_port_rets(stock_rets, port, self.sim_index)
            for i, util in enumerate(utils):
                port_stats = calc_port_path_stats(port_rets, util)
                if i == 0:
                    res['port_exp_rets'][j] = port_stats['exp_ret']
                    res['port_stds'][j] = port_stats['w_std']
                res['port_ce_rets'][i, j] = port_stats['exp_ret_ce']

        idx_ret_paths = gen_port_rets(stock_rets, idx_weights, self.sim_index)
        for i, util in enumerate(utils):
            idx_stats = calc_port_path_stats(idx_ret_paths, util)
            if i == 0:
                res['idx_exp_ret'] = idx_stats['exp_ret']
                res['idx_std'] = idx_stats['w_std']
            res['idx_ce_ret'][i] = idx_stats['exp_ret_ce']

        return res


def main():
    # Set up the simulation environment
    sim = Simulation(data_dir=DATA_DIR, basket=BASKET, returns=RETURNS_DATA,
                     freq=SIM_FREQ, n_paths=N_PATHS, target_ret=TARGET_RET)

    cer_port = pd.DataFrame(index=CRRA_LIST, columns=[STOCKS_IN_PORT_LIST])
    cer_indx = pd.Series(index=CRRA_LIST, dtype=float)

    for stocks_in_port in STOCKS_IN_PORT_LIST:
        prob_of_selection = pd.Series(np.ones(sim.n_stocks) / sim.n_stocks,
                                      index=sim.idx_weights.index)
        ports = gen_sim_ports(N_PORTS, stocks_in_port, prob_of_selection)

        print(f"\nRunning simulation for {N_PORTS} {stocks_in_port}-stock portfolios\n")

        with codetiming.Timer("an_port_set execution"):
            res = sim.an_port_set(ports, CRRA_list=CRRA_LIST)

        print(f"\nResults for {stocks_in_port} stocks in the portfolio:")
        port_stats = {'exp_ret': res['port_exp_rets'].mean(),
                      'std': res['port_stds'].mean(),
                      'ce_ret': {cr: res['port_ce_rets'][i].mean() for i, cr in enumerate(CRRA_LIST)},
                      }
        idx_stats = {'exp_ret': res['idx_exp_ret'],
                     'std': res['idx_std'],
                     'ce_ret': {cr: res['idx_ce_ret'][i] for i, cr in enumerate(CRRA_LIST)}
                     }
        stats = {'port': port_stats, 'idx': idx_stats}
        pprint(stats)

        for cr in CRRA_LIST:
            cer_port.loc[cr, stocks_in_port] = port_stats['ce_ret'][cr]
            cer_indx[cr] = idx_stats['ce_ret'][cr]

    cer_port['Index'] = cer_indx
    print("\nCertainty-Equivalent Returns:")
    print(cer_port)

    cer_port.to_csv(Path(DATA_DIR) / 'cer_port.csv')


if __name__ == "__main__":
    main()
    exit(0)
