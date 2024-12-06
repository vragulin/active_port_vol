""" Demonstrate that undiversified portfolios result in excess vol vs. the index
	V. Ragulin - 11-Sep-2023
"""

import numpy as np
import pandas as pd
import pickle
from pathlib import Path

DATA_DIR = r"/data"
BASKET = 'b500_basket_20231106.csv'
INDEX_HIST = 'idx_long_rets_20231106_vals.csv'
RETURNS_DATA = 'returns_20231106.pickle'
COV_DATA = 'cov_data.pickle'
ANN_FACTOR = 252
LOAD_COV_MATRIX = False
DAILY, WEEKLY = range(2)
COV_MAT_FREQ = WEEKLY
AVERAGE_VARS = False  # Whether to average variances or volatilities
MEDIAN = False  # Use median rather than mean for vol estimation

# Portfolios
REGENERATE_PORTS = True
PORTS = 'portfolios.pickle'

N_PORTS = 10_000
STOCKS_IN_PORT = 100

# Portfolio weight and probability of selection method
EQ_W, MCAP_W = range(2)
EQ_P, MCAP_P = range(2)


def load_data() -> dict:
	data_dict = {}

	if LOAD_COV_MATRIX:
		# Load cov matrix and related data
		data_dict['cov_data'] = pickle.load(open(Path(DATA_DIR) / COV_DATA, 'rb'))

	# Load stock weights
	data_dict['basket'] = pd.read_csv(Path(DATA_DIR) / BASKET, index_col=0)

	# Load stock and index history data
	data_dict['rets'] = pickle.load(open(Path(DATA_DIR) / RETURNS_DATA, 'rb'))

	return data_dict


def gen_sim_ports(n_ports: int, stocks_in_port: int, weights: pd.Series) -> dict:
	""" Generate simulation equal-weighted portfolios, so that their sum matches index weights
	:param n_ports: number of portfolios to simulate
	:param stocks_in_port: number of stocks in each portfolio
	:param weights: target weights of the aggregated portfolio univers
	"""

	rng = np.random.default_rng()
	ports = np.zeros((n_ports, stocks_in_port), dtype=int)
	n_stocks = len(weights)
	w_norm = weights / weights.sum()
	for i in range(n_ports):
		ports[i, :] = rng.choice(n_stocks, size=stocks_in_port, replace=False,
		                         p=w_norm.values)

	# Generate an array of tickers
	def ticker(i: int) -> str:
		return weights.index[i]

	ports_tickers = np.vectorize(ticker)(ports)

	# Calculate actual weights and weight differences
	freqs = pd.DataFrame(np.nan, index=weights.index, columns=['n', 'p', 'p_tgt', 'dp'])
	for i in range(n_stocks):
		freqs.loc[freqs.index[i], 'n'] = np.sum(ports == i)

	freqs.p = freqs.n / n_ports / stocks_in_port
	freqs.p_tgt = w_norm
	freqs.dp = freqs.p - freqs.p_tgt

	out_dict = {'ports_indices': ports,
	            'ports_tickers': ports_tickers,
	            'freqs': freqs}

	return out_dict


def calc_cov_matrix(inp: dict, freq: int = DAILY) -> pd.DataFrame:
	""" Calculate cov matrix from the returns data.  Fill gaps with average var/covar
		:param inp: dictionary with market data
		:return: covariance matrix
	"""

	if freq == DAILY:
		cov_mat = inp['rets']['r_stk'].cov()
	elif freq == WEEKLY:
		tri = (1 + inp['rets']['r_stk']).cumprod()
		tri_weekly = tri[::5]
		ret_weekly = tri_weekly.pct_change().dropna(how='all')
		cov_mat = ret_weekly.cov()
	else:
		raise NotImplementedError(f"Invalid freq: {freq}")

	# Fill gaps with average covar (for off-diag terms) and average corr (diag terms)
	avg_var = np.nanmean(np.diag(cov_mat))
	n_stocks = len(cov_mat)
	avg_covar = (np.nanmean(cov_mat) * n_stocks - avg_var) / (n_stocks - 1)

	cov_dflt = np.identity(n_stocks) * (avg_var - avg_covar) + np.ones((n_stocks, n_stocks)) * avg_covar
	cov_mat = cov_mat.where(~np.isnan(cov_mat), cov_dflt) * ANN_FACTOR / (5 if freq == WEEKLY else 1)

	return cov_mat


def an_port_set(ports: dict, inp: dict, method: int = MCAP_W, scale_factor: float = 1,
                average_vars: bool = False, use_median: bool = False) -> dict:
	""" Analyse a set of portfolios
	:param ports: container of the portfolio set
	:param inp: container with market and index basket data
	:param method: if EQ_W we build equal weighted portfolios, else market-cap weighted
	:param scale_factor: used to translate weights into position size
	:param average_vars: whether to average vars or vols
	:param use_median: whether to use median (not mean) for averaging
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

	# Create a 'dense' representation of portfolios that shows position for each stock
	if 'cov_data' in inp:
		cov_mat = inp['cov_data']['cov_mat'] * ANN_FACTOR
	else:
		cov_mat = calc_cov_matrix(inp, freq=COV_MAT_FREQ)

	n_stocks = len(cov_mat)

	ports_full = np.zeros((n_ports, n_stocks))
	for i in range(n_stocks):
		ports_full[:, i] = (pos_weights * (port_indices == i)).sum(axis=1)

	# Compare aggregate weight vs. the index
	w_agg = ports_full.sum(axis=0)
	df_agg = pd.DataFrame(np.nan, index=cov_mat.index, columns=['w_agg', 'w_tgt', 'd_w'])
	df_agg['w_agg'] = w_agg / w_agg.sum()
	df_agg['w_tgt'] = idx_weights
	df_agg['d_w'] = df_agg.w_agg - df_agg.w_tgt

	# Calculate volatilities for the individual portfolios and for the index
	ports_full_norm = (ports_full / ports_full.sum(axis=1)[:, None])
	# vars = np.diag(ports_full_norm @ cov_mat.values @ ports_full_norm.T)
	vars = np.zeros(n_ports)
	for i in range(n_ports):
		vars[i] = ports_full_norm[i, :].T @ cov_mat.values @ ports_full_norm[i, :]

	if use_median:
		avg_func = np.median
	else:
		avg_func = np.mean

	if average_vars:
		# avg_vol = np.sqrt(vars.mean())
		avg_vol = np.sqrt(avg_func(vars))
	else:
		# avg_vol = np.mean(np.sqrt(vars))
		avg_vol = avg_func(np.sqrt(np.maximum(vars, 0)))

	idx_vol = np.sqrt(df_agg['w_agg'].T @ cov_mat @ df_agg['w_agg'])

	# Calculate the actual b500 vol over the same period
	start_date = inp['rets']['r_stk'].index[0]
	idx_tri = (inp['rets']['r_idx']['cum_tot_ret']).loc[start_date:]
	if COV_MAT_FREQ == DAILY:
		idx_hvol = np.log(idx_tri).diff().std() * np.sqrt(ANN_FACTOR)
	elif COV_MAT_FREQ == WEEKLY:
		idx_hvol = np.log(idx_tri).diff(5).std() * np.sqrt(ANN_FACTOR / 5)
	else:
		raise NotImplementedError(f"Invalid freq: {COV_MAT_FREQ}")

	# Pack results into output dict
	res = {'pos_weights': pos_weights,
	       'ports_full': ports_full,
	       'w_stats': df_agg,
	       'avg_vol': avg_vol,
	       'idx_vol': idx_vol,
	       'idx_hvol': idx_hvol}

	return res


if __name__ == "__main__":
	# Load data
	inp = load_data()

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

	print(f"\nSim settings: avg_var {AVERAGE_VARS}, # stocks {STOCKS_IN_PORT}, # ports {N_PORTS}\n")

	# Base case - equal probability, market-cap weight
	for med_flag, funcname in zip([False, True], ['mean()', 'median()']):
		for avg_var_flag in [True, False]:
			print(f"\nAverage vars = {avg_var_flag}, use {funcname} for averaging:")
			res = an_port_set(ports[EQ_P], inp, method=MCAP_W, average_vars = avg_var_flag, use_median=med_flag)
			print("Equal Prob, Market-Cap weighted Portfolios:")
			print(f"Avg Port Vol = {res['avg_vol'] * 100:.2f}, "
			      f"Agg Basket Vol = {res['idx_vol'] * 100:.2f}, "
			      f"Idx Hist Vol = {res['idx_hvol'] * 100:.2f}")

	# # # Itermediate case - mcap-weighted probability, equal weight
	# print("\nMarket-Cap weighted Prob, Equal weighted Portfolios:")
	# res = an_port_set(ports[MCAP_P], inp, method=EQ_W)
	# print(f"Avg Port Vol = {res['avg_vol'] * 100:.2f}, "
	#       f"Agg Basket Vol = {res['idx_vol'] * 100:.2f}, "
	#       f"Idx Hist Vol = {res['idx_hvol'] * 100:.2f}")
	#
	# # Highest diversification - equal-probability, equal weight
	# print("\nEqual weighted Prob, Equal weighted Portfolios:")
	# res = an_port_set(ports[EQ_P], inp, method=EQ_W)
	# print(f"Avg Port Vol = {res['avg_vol'] * 100:.2f}, "
	#       f"Agg Basket Vol = {res['idx_vol'] * 100:.2f}, "
	#       f"Idx Hist Vol = {res['idx_hvol'] * 100:.2f}")

	print('\nDone')
