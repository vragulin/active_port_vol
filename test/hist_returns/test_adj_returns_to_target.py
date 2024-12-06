import pytest
import pandas as pd
import numpy as np
from src.hist_returns.excess_vol_w_median_hist_ret import adj_returns_to_target


def test_adj_returns_to_target_2stocks():
    # Sample data
    stock_rets = pd.DataFrame({
        'A': [0.01, 0.02, 0.03, 0.04, 0.05],
        'B': [0.02, 0.03, 0.04, 0.05, 0.06]
    }, index=pd.date_range(start='2023-01-01', periods=5, freq='Y'))

    weights = pd.Series([0.6, 0.4], index=['A', 'B'])
    target_ret = 0.1  # 10% annual return

    # Call the function
    adjusted_rets = adj_returns_to_target(stock_rets, weights, target_ret)

    # Expected results (this should be calculated based on the logic of the function)
    expected_rets = pd.DataFrame({
        'A': [0.074568603,
              0.085207896,
              0.095847189,
              0.106486482,
              0.117125775],
        'B': [0.085207896,
              0.095847189,
              0.106486482,
              0.117125775,
              0.127765068]
    }, index=pd.date_range(start='2023-01-01', periods=5, freq='Y'))

    # Use np.testing.assert_array_almost_equal to compare the DataFrames
    pd.testing.assert_frame_equal(adjusted_rets, expected_rets, check_less_precise=True)


def test_adj_returns_to_target_3stocks():
    # Sample data
    stock_rets = pd.DataFrame({
        'A': [0.2, -0.1, 0.3, 0, -0.09],
        'B': [0, 0.1, 0.15, -0.2, 0.05],
        'C': [-0.1, 0, 0, 0.2, -0.05]},
        index=pd.date_range(start='2023-01-01', periods=5, freq='Y'))
    weights = pd.Series([0.2, 0.3, 0.5], index=['A', 'B', 'C'])
    target_ret = 0.05  # 10% annual return

    # Call the function
    adjusted_rets = adj_returns_to_target(stock_rets, weights, target_ret)

    # Expected results (this should be calculated based on the logic of the function)
    expected_rets = pd.DataFrame({
        'A': [0.232443036,
              -0.075667723,
              0.335146622,
              0.027035863,
              -0.065397365],
        'B': [0.027035863,
              0.129739449,
              0.181091242,
              -0.17837131,
              0.078387656],
        'C': [-0.075667723,
              0.027035863,
              0.027035863,
              0.232443036,
              - 0.02431593]
    }, index=pd.date_range(start='2023-01-01', periods=5, freq='Y'))

    # Use np.testing.assert_array_almost_equal to compare the DataFrames
    pd.testing.assert_frame_equal(adjusted_rets, expected_rets, check_less_precise=True)
