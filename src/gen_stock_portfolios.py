"""  Generate a set of equal weighted stock portfolios such that each stock has a given
	 probability of being selected.
 """
import numpy as np
import pandas as pd

if __name__ == "__main__":
	n_stocks = 500
	stocks = np.arange(n_stocks)
	weights = np.ones(n_stocks) / n_stocks
	stocks_in_port = 50
	n_port = 100000
	rng = np.random.default_rng()

	ports = np.zeros((n_port, stocks_in_port), dtype=int)
	for i in range(n_port):
		ports[i, :] = rng.choice(n_stocks, size=stocks_in_port, replace=False, p=weights)

	# print(ports)

	# Check theoretical vs. actual frequencies
	freqs = pd.DataFrame(np.nan, index=range(n_stocks), columns=['n', 'p', 'p_tgt', 'dp'])
	for i in range(n_stocks):
		freqs.loc[i, 'n'] = np.sum(ports == i)

	freqs.p = freqs.n / n_port / stocks_in_port
	freqs.p_tgt = weights
	freqs.dp = freqs.p - freqs.p_tgt

	print(freqs)
	print(freqs.sum())
	print('Done')

	# ToDo: in addition to the volatilities, maybe calcalate a histogram of portfolio returns (since we have actual stock returns over the 10-year period)