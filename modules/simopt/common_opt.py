import numpy as np
import pylab as pl

def get_bounds(guesses, window, d_min, d_max):
    bounds_d = []
    for i in range(len(guesses)):
        if window != None:
            hi = guesses[i]+window/2
            hi = min(d_max, hi)
            lo = guesses[i]-window/2
            lo = max(d_min,lo)
            bounds_d.append((lo, hi))            ###bound for i_th path's distance
        else:
            bounds_d.append((d_min, d_max))      ###bound for i_th path's distance
    return bounds_d

def get_X(pds, divs):
    mat = []
    for d in pds:
        s = np.exp(-2j*np.pi*d/divs)
        mat.append(s)
    mat = np.array(mat).transpose()
    return mat

def get_amps(ds, ch_l1, lambs1):
    x_sub = get_X(ds, lambs1)
    x_pinv = np.linalg.pinv(x_sub)
    amps = np.dot(x_pinv, ch_l1)
    return amps

def get_1ant_chan(ds, ch_l1, lambs1, lambs2):
    x_sub = get_X(ds, lambs1)
    x_pinv = np.linalg.pinv(x_sub)
    amps = np.dot(x_pinv, ch_l1)
    amps = amps/sum(np.abs(amps))
    # print np.abs(amps)
    # print ds

    x_sub = get_X(ds, lambs2)
    y2 = np.dot(x_sub, amps)

    return y2.ravel()