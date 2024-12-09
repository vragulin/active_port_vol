"""  Simulate what would the analysis look like for completely random returns, as the baseline for comparison
    V.Ragulin   8-Dec-2023
"""
import numpy as np
import pandas as pd
from sys import exit

# Global variables
N_STOCKS = 500
STOCKS_IN_PORT = 500
STK_RET = 0.10
STOCK_VOL = 0.3
WEIGHTS_VOL = 2
N_PATHS = 100000
N_STEPS = 52
RNG = np.random.default_rng(123)


def main():
    # Generate Random weights
    weights = np.exp(RNG.normal(0, WEIGHTS_VOL, N_STOCKS))
    weights /= weights.sum()

    # Generate random returns
    STK_LOG_RET =STK_RET - 0.5 * STOCK_VOL ** 2
    lret_1step = STK_LOG_RET / N_STEPS
    vol_1step = STOCK_VOL / np.sqrt(N_STEPS)

    wealth_final = np.zeros(N_PATHS)
    for i in range(N_PATHS):
        port_weights = RNG.choice(weights, size=STOCKS_IN_PORT, replace=False)
        port_weights /= port_weights.sum()
        stock_rets = np.exp(RNG.normal(lret_1step, vol_1step, (N_STEPS,STOCKS_IN_PORT))) -1
        port_rets = stock_rets @ port_weights
        wealth_final[i] = np.prod(1 + port_rets)

    # Calculate statistics
    exp_ret = np.mean(wealth_final) - 1
    exp_lret = np.mean(np.log(wealth_final))
    std = np.std(np.log(wealth_final))
    print(f"Expected return: {exp_ret:.4f}, Expected log return: {exp_lret:.4f}, Std: {std:.4f}")


if __name__ == '__main__':
    main()
    exit(0)
