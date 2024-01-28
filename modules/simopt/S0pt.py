import numpy as np
from scipy.optimize import minimize, differential_evolution, basinhopping, least_squares, brute
from mystic.solvers import fmin_powell
import common_opt
import bisect

import sys
from os import path

sys.path.append("../")
sys.path.append("../")

from .. import datasetGen, rfCommon
import pylab as pl


def fit_simple(ch_l1, l1, l2, sep, min_d, max_d, min_meth, init_guesses, window):
    K = ch_l1.shape[1]
    distances = np.arange(min_d, max_d, 1)
    guessed_ch_l1 = np.zeros(ch_l1.shape).astype(np.complex)
    guessed_ch_l2 = np.zeros([len(l2), K]).astype(np.complex)

    bounds_d = common_opt.get_bounds(init_guesses, window, min_d, max_d)

    for k in range(K):
        guesses_S, errors = solve_opt_O2(init_guesses, l1, ch_l1[:, k], min_d, max_d, min_meth, bounds_d)
        guessed_ch_l1[:, k] = common_opt.get_1ant_chan(guesses_S, ch_l1[:, k], l1, l1)
        guessed_ch_l2[:, k] = common_opt.get_1ant_chan(guesses_S, ch_l1[:, k], l1, l2)

    return guessed_ch_l1, guessed_ch_l2


def fit_simple_K0(ch_l1, l1, l2, sep, min_d, max_d, min_meth, init_guesses, window):
    K = ch_l1.shape[1]
    distances = np.arange(min_d, max_d, 1)
    guessed_ch_l1 = np.zeros(ch_l1.shape).astype(np.complex)
    guessed_ch_l2 = np.zeros(ch_l1.shape).astype(np.complex)

    bounds_d = common_opt.get_bounds(init_guesses, window, min_d, max_d)

    for k in range(K):
        if k > 0:
            bounds_d = common_opt.get_bounds(guesses_S, 2 * k * sep, min_d, max_d)
        guesses_S, errors = solve_opt_O2(init_guesses, l1, ch_l1[:, k], min_d, max_d, min_meth, bounds_d)
        guessed_ch_l1[:, k] = common_opt.get_1ant_chan(guesses_S, ch_l1[:, k], l1, l1)
        guessed_ch_l2[:, k] = common_opt.get_1ant_chan(guesses_S, ch_l1[:, k], l1, l2)

    return guessed_ch_l1, guessed_ch_l2


def fit_simple_bestK(ch_l1, l1, l2, sep, min_d, max_d, min_meth, init_guesses, window):
    K = ch_l1.shape[1]
    distances = np.arange(min_d, max_d, 1)
    guessed_ch_l1 = np.zeros(ch_l1.shape).astype(np.complex)
    guessed_ch_l2 = np.zeros(ch_l1.shape).astype(np.complex)

    bounds_d = common_opt.get_bounds(init_guesses, window, min_d, max_d)

    min_error = 10000
    best_guesses = []
    best_K = 0
    for k in range(K):
        guesses_S, errors = solve_opt_O2(init_guesses, l1, ch_l1[:, k], min_d, max_d, min_meth, bounds_d)
        if errors[-1] < min_error:
            min_error = errors[-1]
            best_guesses = guesses_S
            best_K = k
        guessed_ch_l1[:, k] = common_opt.get_1ant_chan(guesses_S, ch_l1[:, k], l1, l1)
        guessed_ch_l2[:, k] = common_opt.get_1ant_chan(guesses_S, ch_l1[:, k], l1, l2)

    for k in range(K):
        k_sep = abs(k - best_K)
        bounds_d = common_opt.get_bounds(guesses_S, 2 * k_sep * sep, min_d, max_d)
        guesses_S, errors = solve_opt_O2(init_guesses, l1, ch_l1[:, k], min_d, max_d, min_meth, bounds_d)
        guessed_ch_l1[:, k] = common_opt.get_1ant_chan(guesses_S, ch_l1[:, k], l1, l1)
        guessed_ch_l2[:, k] = common_opt.get_1ant_chan(guesses_S, ch_l1[:, k], l1, l2)

    return guessed_ch_l1, guessed_ch_l2


def solve_opt_O2(guesses, lambs, Y, d_min, d_max, meth, bounds_d=None):
    progress_list = []
    opt_O_arg_list = tuple([lambs, Y, progress_list])
    if bounds_d == None:
        bounds_d = common_opt.get_bounds(guesses, None, d_min, d_max)

    if meth == "fmin":
        sol = fmin_powell(opt_O2, guesses, bounds=bounds_d, args=opt_O_arg_list, disp=0, ftol=1e-36, xtol=1e-33)
        if len(guesses) == 1:
            sol = [sol]
    elif meth == "min":
        sol = minimize(opt_O2, guesses, bounds=bounds_d, args=opt_O_arg_list, method="SLSQP",
                       options={"maxiter": 100000, "ftol": 1e-8})
        sol = sol.x
    elif meth == "basin":
        sol = basinhopping(opt_O2, guesses, minimizer_kwargs={"args": opt_O_arg_list}, niter=10, niter_success=500)
        sol = sol.x
    elif meth == "diff":
        sol = differential_evolution(opt_O2, args=opt_O_arg_list, bounds=bounds_d)
        sol = sol.x
    elif meth == "brute":
        print
        "brute", bounds_d
        sol = brute(opt_O2, args=opt_O_arg_list, ranges=bounds_d)
        print
        sol
    elif meth == "lst":
        sol = least_squares(opt_O2, guesses, args=opt_O_arg_list, bounds=(d_min, d_max), loss="soft_l1")
        sol = sol.x
    else:
        print
        "unknown minimization method requested:", meth
    return sol, progress_list


def opt_O2(guesses, lambs, Y, progress_list):
    amp_err = 0.0
    X_sub = common_opt.get_X(guesses, lambs)
    inv_X_sub = np.linalg.pinv(X_sub)
    amp = np.dot(inv_X_sub, Y)

    Y_bar = np.dot(X_sub, amp)

    O = np.mean(np.abs(Y - Y_bar) ** 2)
    # O = np.mean(np.abs(Y - Y_bar))**2
    # O = np.linalg.norm(Y - Y_bar)**2

    amp = np.abs(amp)
    amp_err = 0.001 * np.sum(amp)

    O = O + amp_err

    progress_list.append(O)
    return O


def get_error(guesses, lambs, Y):
    return opt_O2(guesses, lambs, Y, [])
