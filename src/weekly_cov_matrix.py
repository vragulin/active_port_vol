""" Generate historical covariance matrix from weekly data
	to be used for calculating portfolio risk
"""
import pickle
from pathlib import Path

DATA_DIR = r'../data_5y'
BASKET = 'b500_basket_20231106.csv'
INDEX_HIST = 'idx_long_rets_20231106_vals.csv'
RETURNS_DATA = 'returns_20231106.pickle'
COV_DATA = 'cov_data.pickle'
ANN_FACTOR = 252


def main():
	# Load cov matrix and related data
	cov_data = pickle.load(open(Path(DATA_DIR) / COV_DATA, 'rb'))

	# Load stock and index history data
	ret_data = pickle.load(open(Path(DATA_DIR) / RETURNS_DATA, 'rb'))

	print("Done")


if __name__ == "__main__":
	main()
