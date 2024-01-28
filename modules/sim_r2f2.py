
import os, time, sys
sys.path.insert(0, "/home/avishek/PycharmProjects/Optml_avishek/modules")
import numpy as np
from modules.r2f2_core import r2f2
from modules.r2f2_core import r2f2_helper
from modules import rfCommon

def run_simulation_test(ch_l1, l1, l2, K, sep, max_num_paths, max_d, min_sep, full_exp_dict):
    initial_guesses = r2f2_helper.get_initial_guesses(l1, K, sep, ch_l1, full_exp_dict, max_d, psi_step=1, dist_step=1, show_plots=False)
    best_sol, reason = r2f2.r2f2_solver(ch_l1, l1, sep, max_d, initial_guesses, max_num_paths)
    d_ns = best_sol[0::2]
    psi_ns = best_sol[1::2]
    param_guesses = r2f2.get_full_params_method3(d_ns, psi_ns, K, sep, l1, ch_l1)
    c_l1 = rfCommon.get_chans_from_params(param_guesses, K, sep, l1)
    c_l2 = rfCommon.get_chans_from_params(param_guesses, K, sep, l2)
    return c_l2, len(d_ns)
