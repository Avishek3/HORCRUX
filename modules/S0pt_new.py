import numpy as np
from scipy.optimize import minimize, differential_evolution, basinhopping, least_squares, brute
from mystic.solvers import fmin_powell
import common_opt

import datasetGen, rfCommon
import matplotlib.pyplot as plt


def pred(model, K, ch_l1, l1, l2, sep, min_d, max_d, min_meth, window, distances, do_plot):
    rots = np.arange(-1 * sep, sep, sep / 10)
    tots = np.zeros(len(distances))
    dats = []
    th = 0.4#0.4#0.4#0.5
    dat = datasetGen.to_reals(ch_l1[:, 0]).transpose()
    g = model.predict(dat)[0]
    g = g / np.max(g)
    g = np.where(g < th, 0, g)
    ds = rfCommon.get_peak_distances(g, distances)
    print("NNDE distances:", np.round(ds))

    if len(ds) == 0:
        return None, 0

    return ds