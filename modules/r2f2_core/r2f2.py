import numpy as np
import pylab as pl
from mystic.solvers import fmin_powell
from scipy.optimize import minimize
from modules.r2f2_core import r2f2_helper
from numpy.linalg import lstsq


###########################################################
###     R2F2                                            ###
###########################################################
def get_F_KK(sep, K):
    '''
    get non uniform fourier matrix
    sep = inter antenna separation
    K = num antennae

    though the equation reduces to be independent of sep:
    orig = -2pi i l j_prime psi_bar/lambda
    psi_bar = lambda/L
    L = K*l
    orig becomes --> -2pi i j_prime/K
    '''

    mat = np.zeros([K, K]).astype(np.complex128)
    for i in range(K):
        for j_prime in range(K):
            mat[i, j_prime] = np.exp(-2j * np.pi * i * j_prime / K)
    mat = mat / K
    return mat
    # For one antenna case, the result is always, array([[1.+0.j]])


def get_P(Finv, h_IK):
    '''
    F_KK = KxK non-uniform fourier matrix
    h_IK = IxK channels for K antennae and I freqs(lambs)
    '''
    I = h_IK.shape[0]
    K = h_IK.shape[1]
    P = np.zeros([I * K, 1]).astype(np.complex128)
    P = np.matrix(P)
    for i in range(I):
        h_i = np.matrix(h_IK[i, :]).transpose()
        x = np.dot(Finv, h_i)
        P[i * K:(i + 1) * K, 0] = x
    return P


def get_Sincs(L, all_lambs, K, N, psi_js):
    '''
    L = antenna array length
    all_lambs = list of wavelengths
    K = num antennae
    N = num paths
    psi_js = phi for each path

    after correspondence with the 1st author, I've implemented the
    S matrix as intended by them. The sinc has an additional
    complex part.
    '''
    S = []
    scale = (K - 1.0) / float(K)
    for lamb in all_lambs:
        S_lamb = np.ones([K, N]).astype(np.complex128)
        psi_bar = lamb / L
        for i in range(K):
            for j in range(N):
                v = np.pi * (i * psi_bar - psi_js[j]) * L / lamb
                denominator = np.sin(v / K)
                if denominator == 0:
                    x = 1
                else:
                    x = np.sin(v) / denominator
                    x = x * np.exp(1j * np.pi * scale * (i * psi_bar - psi_js[j]) / psi_bar)  ##complex part
                S_lamb[i, j] = x
        S.append(S_lamb)
    return S


def get_D(N, all_lambs, d_ns):
    '''
    N = num paths
    all_lambs = list of wavelengths
    d_ns = list of distances of each of N paths
    '''
    D_is = []
    I = len(all_lambs)
    for i in range(I):
        D_i = np.zeros([N, N]).astype(np.complex128)
        for k in range(N):
            D_i[k, k] = np.exp(-2j * np.pi * d_ns[k] / all_lambs[i])
        D_is.append(D_i)
    return D_is


def get_SD(L, K, N, all_lambs, d_ns, psi_js):
    S_list = get_Sincs(L, all_lambs, K, N, psi_js)
    D_list = get_D(N, all_lambs, d_ns)
    for i in range(len(all_lambs)):
        S_i = S_list[i]
        D_i = D_list[i]
        SD_i = np.dot(S_i, D_i)
        if i == 0:
            SD = SD_i
        else:
            SD = np.vstack([SD, SD_i])
    return SD


def opt_O(initial_guesses, P, L, K, N, all_lambs, sep, progress_list):
    d_ns = initial_guesses[0::2]
    psi_js = initial_guesses[1::2]
    S = get_SD(L, K, N, all_lambs, d_ns, psi_js)
    Spinv = np.linalg.pinv(S)
    O = np.linalg.norm(P - np.dot(np.dot(S, Spinv), P)) ** 2
    progress_list.append(O)
    return O


def get_error(initial_guesses, L, K, N, all_lambs, sep, H_IK):
    F = get_F_KK(sep, K)
    Finv = np.linalg.inv(F)
    P = get_P(Finv, H_IK)
    d_ns = initial_guesses[0::2]
    psi_js = initial_guesses[1::2]
    S = get_SD(L, K, N, all_lambs, d_ns, psi_js)
    Spinv = np.linalg.pinv(S)
    error = np.linalg.norm(P - np.dot(np.dot(S, Spinv), P)) ** 2
    return error


def solve_opt_O(guesses, L, K, N, all_lambs, sep, H_IK, max_distance):
    '''
    returns distance and psi values after solving
    optimization problem
    '''
    F = get_F_KK(sep, K)
    Finv = np.linalg.inv(F)
    P = get_P(Finv, H_IK)
    progress_list = []
    opt_O_arg_list = tuple([P, L, K, N, all_lambs, sep, progress_list])
    bounds_x = []
    for i in range(N):
        bounds_x.append((0, max_distance))  ###bound for i_th path's distance
        bounds_x.append((-1, 1))  ###bound for i_th path's psi

    sol = fmin_powell(opt_O, guesses, bounds=bounds_x, args=opt_O_arg_list, disp=0)

    # sol = minimize(opt_O, guesses, bounds=bounds_x, args=opt_O_arg_list, method="SLSQP")
    # sol = sol.x

    return sol, progress_list


def get_full_params_method1(d_ns, psi_ns, K, sep, input_lambs, input_chan):
    '''
    Find the other parameters (a_js and phi_js) to complete the
    4-tuple needed to represent the physical paths

    returns the 4-tuples

    Solves for attenuations and phis based on eq 4 of the R2F2 paper
    '''
    L = K * sep
    n_paths = len(d_ns)
    F = get_F_KK(sep, K)

    ###solve for only 1 wavelength, same tuples for all
    ###solving for 0th wavelength
    wl_in = input_lambs[0]
    S_i = get_Sincs(L, [wl_in], K, n_paths, psi_ns)[0]

    h_i = np.array(input_chan[0, :])
    X = np.dot(F, S_i)
    X = np.linalg.pinv(X)
    a_vect_input = np.dot(X, h_i)
    a_ns = np.abs(a_vect_input)
    a_ns = a_ns / np.sum(a_ns)

    exp_d_ns = np.exp(-2j * np.pi * np.array(d_ns) / wl_in)
    phi_ns = a_vect_input / (a_ns * exp_d_ns)
    phi_ns = np.angle(phi_ns)

    return d_ns, a_ns, phi_ns, psi_ns


def get_full_params_method2(d_ns, psi_ns, K, sep, input_lambs, input_chan):
    '''
    Find the other parameters (a_js and phi_js) to complete the
    4-tuple needed to represent the physical paths

    returns the 4-tuples

    Solves based on the equation for initial guesses. Gets contribution
    of a path to the observed channel.
    '''
    L = K * sep
    n_paths = len(d_ns)
    F = get_F_KK(sep, K)

    ###solve for only 1 wavelength, same tuples for all
    ###solving for 0th wavelength
    a_ns = []
    phi_ns = []
    for i in range(n_paths):
        params_i = [[d_ns[i]], [1.0], [0], [psi_ns[i]]]
        ch = get_chans_from_params(params_i, K, sep, input_lambs)
        ch = np.conjugate(ch)
        ch = ch * input_chan
        ch = np.sum(ch)
        a = np.abs(ch)
        phi = np.angle(ch)
        a_ns.append(a)
        phi_ns.append(phi)
    a_ns = np.array(a_ns)
    a_ns = a_ns / np.sum(a_ns)
    return d_ns, a_ns, phi_ns, psi_ns


def get_full_params_method3(d_ns, psi_ns, K, sep, input_lambs, input_chan):
    '''
    Find the other parameters (a_js and phi_js) to complete the
    4-tuple needed to represent the physical paths

    returns the 4-tuples

    Given the distances [d1,d2,...] that the signal comes from, make a
    matrix X in which each column represents the channel from a single
    path of length d. Solve Y = Xa, where Y is the channel observed at
    the 0th antenna, and a is a complex vector which contains the
    attenuations and phis of each path.
    '''
    n_paths = len(d_ns)
    X = []
    for i in range(n_paths):
        d = d_ns[i]
        l = np.exp(-2j * np.pi * d / input_lambs)
        X.append(l)
    X = np.array(X).transpose()
    Y = input_chan[:, 0]  ##channel at 0th antenna
    a_complex = lstsq(X, Y, rcond=None)[0]
    a_ns = np.abs(a_complex)
    a_ns = np.array(a_ns)
    a_ns = a_ns / np.sum(a_ns)

    phi_ns = np.angle(a_complex)

    return d_ns, a_ns, phi_ns, psi_ns


def get_chans_from_params(params, K, sep, lambs):
    '''
    given the params/4-tuples for the physical paths, compute the channel
    across K antenna for given wavelengths.
    Based on Equation 3 of paper
    '''
    d_ns, a_ns, phi_ns, psi_ns = params
    N = len(d_ns)
    I = len(lambs)
    H = np.zeros([I, K]).astype(np.complex128)

    for i_wl in range(len(lambs)):
        wl = lambs[i_wl]
        ### based on eq 3
        for i_K in range(K):  ###for each antenna
            t = 0
            for i_N in range(N):  ###for each path
                c1 = a_ns[i_N] * np.exp((-2j * np.pi * d_ns[i_N] / wl) + 1j * phi_ns[i_N])
                c1 = c1 * np.exp(-2j * np.pi * (i_K) * sep * psi_ns[i_N] / wl)
                H[i_wl, i_K] += c1
    return H


def r2f2_solver(ul_H_IK, lambs_ul, sep, max_dist, guesses, max_num_paths):
    I = len(lambs_ul)
    K = ul_H_IK.shape[1]
    L = sep * K

    epsilon = 0.01 * I * K
    # epsilon = 0.00001
    perc_th = 0.0001

    ### thresholds for discarding similar paths
    th_psi = 0.3  ##radians
    th_d = 5  ##meters

    d_psis = r2f2_helper.fix_conditioning(guesses, max_dist, th_d, th_psi)

    # max_num_paths = len(d_psis)/2
    max_num_paths = min(max_num_paths, len(d_psis) // 2) + 1
    min_num_paths = 1
    # min_num_paths = max_num_paths-1
    best_valid_solution = d_psis
    old_min_val = new_min_val = 100000000

    reason = []

    for N in range(min_num_paths, max_num_paths):
        top_N = d_psis[:2 * N]
        new_sol, progress_list = solve_opt_O(top_N, L, K, N, lambs_ul, sep, ul_H_IK, max_dist)
        new_min_val = get_error(new_sol, L, K, N, lambs_ul, sep, ul_H_IK)
        # print "Iter:",N, "of",str(max_num_paths), "error:", new_min_val
        # print "Iter: with guesses:", top_N, ", score:", new_min_val
        # print "Iter: final sol:", new_sol

        if not r2f2_helper.sol_is_valid(new_sol):
            # print "BREAK:invalid sol"
            reason = ("invalid_sol", old_min_val)
            break

        if new_min_val > old_min_val:
            # print "SKIP SOL: increase in min val"
            continue

        gain = (abs(old_min_val) - abs(new_min_val)) / new_min_val
        if gain < perc_th:
            # print "BREAK: gain too low"
            reason = ("low_gain", new_min_val)
            break

        new_sol = r2f2_helper.fix_conditioning(new_sol, max_dist, th_d, th_psi)
        # print "Iter: final sol+fix cond.:", new_sol

        best_valid_solution = new_sol

        if new_min_val < epsilon:
            # print "BREAK: minimized below epsilon"
            reason = ("below_epsilon", new_min_val)
            break
        old_min_val = new_min_val

    if len(reason) == 0:
        reason = ("max_paths:" + str(N), new_min_val)

    return best_valid_solution, reason
