import numpy as np
import pylab as pl
from scipy import ndimage
import operator
import random


###########################################################
###     Generate initial guesses                        ###
###########################################################
def precomputer(lambs, K, l, max_d, d_step=1.0, psi_step=1.0):
    exp1, exp2 = get_exps(lambs, K, l)
    full_exp_dict = {}

    thetas = np.arange(0, 180, psi_step)
    thetas_rad = thetas * np.pi / 180
    psis = np.cos(thetas_rad)

    distances = np.arange(0, max_d, d_step)
    for d in distances:
        exp_d = exp1 * d
        for psi in psis:
            exp_psi = exp2 * psi
            key = (d, psi)
            full_exp_val = np.exp(exp_d + exp_psi)
            full_exp_dict[key] = full_exp_val

    return full_exp_dict


def get_exps(lambs, K, l):
    '''
    precompute part of the exponents for the
    fixed parts of the initialization equation
    '''
    I = len(lambs)
    base = np.ones([I, K])
    lamb_i = np.transpose(base) / lambs
    lamb_i = np.transpose(lamb_i)
    k_mat = base * range(0, K)

    exp1 = 2j * np.pi * lamb_i
    exp2 = 2j * np.pi * lamb_i * k_mat * l

    return exp1, exp2


def get_2D_guess_heatmap(psis, distances, full_exp_dict, h_mat):
    psi_d_mat = np.zeros([len(psis), len(distances)]).astype(np.complex128)
    i = 0
    for psi in psis:
        j = 0
        for d in distances:
            key = (d, psi)
            exp_full = full_exp_dict[key]
            val = h_mat * exp_full
            val = np.sum(val)
            psi_d_mat[i, j] = val
            j += 1
        i += 1
    psi_d_mat = np.abs(psi_d_mat)
    psi_d_mat = np.square(psi_d_mat)
    return psi_d_mat


# lambs: wavelengths
# K: antenna #
# l: inter-antenna seperation
# h_mat: channel matrix (?)
# full_exp_dict: a dictionary, with keys as (distance, psi), and values as the expection of a path like the key
# max_dist: maximum possible distance considered for the multi-paths
def get_initial_guesses(lambs, K, l, h_mat, full_exp_dict, max_dist, psi_step=1.0, dist_step=1.0, show_plots=True):
    '''
    R2F2 Section 5.3 initialization
    exponent is broken into 2 parts and partially
    precomputed
    '''
    psi_step = float(psi_step)

    distances = np.arange(0, max_dist, dist_step)

    thetas = np.arange(0, 180, psi_step)
    thetas_rad = thetas * np.pi / 180
    psis = np.cos(thetas_rad)
    psi_d_mat = get_2D_guess_heatmap(psis, distances, full_exp_dict, h_mat)

    if show_plots:
        pl.subplot(2, 2, 1)
        pl.imshow(psi_d_mat, aspect="auto")
        skip = len(thetas_rad) // 20
        y_pos = range(0, len(thetas_rad), skip)
        y_names = np.round(psis[y_pos], 2)
        # y_names = np.round(thetas[y_pos],2)
        pl.yticks(y_pos, y_names)
        pl.colorbar()
        pl.tight_layout()
    # pl.figure()

    p_d_psi = get_peaks(psi_d_mat, distances, psis, show_plots)

    ####sort guesses by "probability"
    p_d_psi = sorted(p_d_psi.items(), key=operator.itemgetter(1), reverse=True)
    p_d_psi = [k for k, v in p_d_psi]

    temp = []
    list(temp.extend(item) for item in p_d_psi)
    p_d_psi = temp

    return p_d_psi


def get_peaks(psi_d_mat, distances, psis, show_plots):
    r, c = psi_d_mat.shape
    win = 5

    y = np.array(psi_d_mat)

    th = ndimage.filters.gaussian_filter(y, win)
    y = np.where(y < th, 0, y)
    if show_plots:
        pl.subplot(2, 2, 2)
        pl.imshow(y, aspect="auto")
        pl.colorbar()

    filt = ndimage.filters.maximum_filter(y, win)
    th = np.mean(y)
    filt = np.where(filt <= th, th, filt)

    y = y / filt

    if show_plots:
        pl.subplot(2, 2, 3)
        pl.imshow(y, aspect="auto")
        pl.colorbar()

    th = 1
    y = np.where(y >= th, 1, 0)

    if show_plots:
        pl.subplot(2, 2, 4)
        pl.imshow(y, aspect="auto")
        pl.colorbar()
        pl.show()

    i_psis, i_distances = np.where(y == 1)

    new_p_d_psi = {}
    for i in range(len(i_psis)):
        key = (distances[i_distances[i]], psis[i_psis[i]])
        new_p_d_psi[key] = psi_d_mat[i_psis[i], i_distances[i]]
    P_d_psi = new_p_d_psi
    return P_d_psi


###########################################################################
######          Other helper functions                  ###################
###########################################################################

def fix_conditioning(ordered_guesses, max_d, th_d, th_psi):
    '''
    ordered_guesses = [d1,psi1,d2,psi2,....dn,psin]
    '''
    keep = []
    keep.append(ordered_guesses[0])
    keep.append(ordered_guesses[1])
    num_guesses = len(ordered_guesses) // 2
    for i in range(num_guesses):
        keep_i = True
        d_i, psi_i = ordered_guesses[2 * i], ordered_guesses[2 * i + 1]
        for j in range(len(keep) // 2):
            d_j, psi_j = keep[2 * j], keep[2 * j + 1]
            d_diff = abs(d_i - d_j)
            # d_diff = min(d_diff, abs(d_diff-max_d))     ### d-space is circular. 0 == max_distance, kinda

            psi_diff1 = abs(psi_i - psi_j)
            psi_diff2 = abs(psi_j - psi_i)
            psi_diff = min(psi_diff1, psi_diff2)
            if psi_diff < th_psi and d_diff < th_d:
                # if d_diff < th_d :
                keep_i = False
                break
        if keep_i:
            keep.extend([d_i, psi_i])
    return keep


def sol_is_valid(sol):
    for i in range(1, len(sol), 2):
        if sol[i] < -1:
            return False
        if sol[i] > 1:
            return False
    return True