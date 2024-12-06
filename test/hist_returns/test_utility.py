import pytest as pt
import numpy as np
import utility as ut

# Global Parameters
RTOL = 1e-6


@pt.mark.parametrize("param, expected", [
    (1, 0),
    (2, 0),
    (3, 0)
])
def test_u_of_wealth_wealth_1(param, expected):
    u = ut.Utility(param=param)
    wealth = 1
    assert u.u_of_wealth(wealth) == expected, f"Error: expected {expected}, got {u.u_of_wealth(wealth)}"


@pt.mark.parametrize("param, expected", [
    (1, -0.105360516),
    (2, -0.111111111),
    (3, -0.117283951)
])
def test_u_of_wealth_wealth_0_9(param, expected):
    u = ut.Utility(param=param)
    wealth = 0.9
    assert pt.approx(u.u_of_wealth(wealth), rel=RTOL
                     ) == expected, f"Error: expected {expected}, got {u.u_of_wealth(wealth)}"


@pt.mark.parametrize("w", [0.9, 1, 1.1])
@pt.mark.parametrize("param", [1, 2, 3])
def test_implied_wealth(w, param):
    u = ut.Utility(param=param)
    u_val = u.u_of_wealth(w)
    imp_w = u.implied_wealth(u_val)
    assert pt.approx(imp_w, abs=RTOL) == w, f"Error: expected {w}, got {imp_w}"


@pt.mark.parametrize("param", [1,2,3])
def test_exp_util(param):
    w = np.array([0.8, 1, 1.2])
    u = ut.Utility(param=1)
    actual = u.expUtil(w)
    expected = np.mean([u.u_of_wealth(wealth) for wealth in w])
    assert pt.approx(actual, abs=RTOL) == expected, f"Error: expected {expected}, got {actual}"


@pt.mark.parametrize("w_arr, param, expected", [
    ((0.8, 1, 1.2), 1, 0.98648483),
    ((0.8, 1, 1.2), 2, 0.97297297),
    ((0.8, 1, 1.2), 3, 0.95974410)
])
def test_ce_wealth(w_arr, param, expected):
    u = ut.Utility(param=param)
    actual = u.ce_wealth(w_arr)
    assert pt.approx(actual, abs=RTOL) == expected, f"Error: expected {expected}, got {actual}"

